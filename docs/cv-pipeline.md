# Computer-vision pipeline

Package: `backend/rallycv` (independent of the web app). Adapter to the API:
`backend/app/analysis/cv_provider.py`.

## Input assumptions

- **Fixed broadcast camera** behind and above one baseline, whole court in
  frame (standard BWF TV view). Hand-held or panning footage is not supported:
  the court is calibrated once per clip.
- **Singles.** One player per half; the near player is "A", the far player "B".
- Ideally **one rally** per clip. Dead time and several rallies are handled
  (see *Rallies*), but replays and camera cuts are skipped, not analysed.

## Stages

| # | Stage | Module | Method |
|---|-------|--------|--------|
| 0 | Detecting court | `court.py`, `camera.py` | Median of 25 frames removes players. Floor region from its dominant colour; white-line pixels (thin, bright, unsaturated); Hough lines; exhaustive fit of the BWF court model scored by *pixels of painted line explained*; least-squares refinement. Focal length and pose recovered from the homography (principal point at image centre), giving a full 3D camera. |
| 1 | Tracking players and shuttle | `players.py`, `tracknet.py`, `shuttle.py` | One streaming pass. Repeated frames (30 fps content in 60 fps files) and off-view frames (replays) are skipped. **YOLO11n** person detection at 15 Hz; feet mapped to the floor; people outside the court (umpires, judges) dropped; near half = A, far half = B. **TrackNetV3** (pretrained, MIT) detects the shuttle on every distinct frame. Without TrackNet weights a classical three-frame-difference detector is used. |
| 2 | Detecting shots | `shots.py` | Detections are linked into tracklets and the best non-overlapping chain is kept. **Hits** are direction-change cusps of the image track within a player's reach (box widened, extended above the head), plus track restarts next to a player. **Landings**: the shuttle falling to rest on the floor; this ends the rally and the next 1.2 s is dead time. Hitters alternate; the first hit of a rally is the serve. |
| 3 | Mapping to court | `mapping.py`, `pipeline.py` | Each shot runs from the hitter's floor position to the receiver's (or the landing spot). In-flight positions come from intersecting camera rays with the vertical plane between hitter and receiver; where that is ill-conditioned (shots straight down the camera axis) progress is interpolated with a drag-like profile. Shot type is a heuristic from start/end depth, speed and apex height. |

The contract output also carries the shuttle's detected **image position**
and the **court→image homography**, so the UI can overlay what was detected
on the video. That overlay is the quickest way to see whether an analysis is
correct.

## Measured accuracy

**Synthetic frames and rallies** (exact ground truth, `tests/cv/`):

- court corners within 4 px at 1080p across three camera set-ups
- focal length within 5%; camera position within 0.6 m; net-post tops within 6 px
- hits found within 0.15 s, attributed to the right player
- landing detected; hit points within 0.5 m; landing spot within 0.3 m

**Real broadcast footage** (China Open 2025, 22 s, 1080p, hand-checked frame by frame):

| Measure | Result |
|---|---|
| Court fit | lines within a few px; net-post height predicted within ~6 px |
| Players | tracked in ~100% of frames, each in the correct half |
| Shuttle | detected in ~75% of distinct frames (TrackNet) |
| Hits | 14 detected; ~12 clearly correct, 1 about 0.5 s late, ~4 missed during long detection gaps |
| Rally structure | landing detected; dead time after it ignored; next serve found |
| Speed | ~35–40 s for a 22 s 1080p60 clip on an Apple M2 (MPS) |

Shot-type labels are heuristic and the least reliable output.

## Known limitations / next steps

1. **Height ambiguity.** A single camera can't resolve an airborne shuttle's
   depth along the court. Floor positions between hits are estimates; hit and
   landing points are reliable.
2. **Missed hits** when the shuttle is undetected for a long stretch. A
   physics-based 3D trajectory fit between hits, or TrackNetV3's InpaintNet,
   would fill these.
3. **Shot types.** A small learned classifier (e.g. trained on ShuttleSet
   labels) would replace the heuristics.
4. **Doubles** and moving cameras are out of scope.
5. **Licensing.** YOLO is AGPL-3.0; see `THIRD_PARTY_NOTICES.md`.

## Running

```sh
make models                       # downloads YOLO + TrackNet weights
make dev-backend                  # CV analyzer (default)
make dev-backend-mock             # simulated data, no models
make test-cv                      # slow end-to-end test on data/samples/rally_china1.mp4
```

Environment variables: `RALLYREVIEW_ANALYZER` (`cv` | `mock`),
`RALLYREVIEW_MODELS_DIR`, `RALLYREVIEW_DATA_DIR`, `RALLYREVIEW_MAX_UPLOAD_BYTES`.

Recommended sample (one complete rally, serve to landing, 21 s): `rally_clean.mp4`.
To recreate the sample clips:

```sh
curl -L -o backend/data/samples/china_open_2025.webm \
  "https://upload.wikimedia.org/wikipedia/commons/2/2f/Wang_Zhiyi_triumph_in_all-Chinese_China_Open_finals_I_China_Open_2025.webm"
ffmpeg -ss 2.2 -t 22 -i backend/data/samples/china_open_2025.webm -c:v libx264 -crf 18 -an \
  backend/data/samples/rally_china1.mp4
ffmpeg -ss 6.0 -t 21 -i backend/data/samples/china_open_2025.webm -c:v libx264 -crf 18 -an \
  -pix_fmt yuv420p -movflags +faststart backend/data/samples/rally_clean.mp4
```
