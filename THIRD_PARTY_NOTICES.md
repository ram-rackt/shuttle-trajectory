# Third-party notices

## TrackNetV3 (shuttle detection)

`backend/rallycv/tracknet.py` contains a re-implementation of the TrackNet
network from https://github.com/qaz812345/TrackNetV3, and the pipeline loads
the authors' pretrained checkpoint (downloaded by
`backend/scripts/download_models.py`).

    MIT License

    Copyright (c) 2024 qaz812345

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Paper: Chen, Y.-J., & Wang, Y.-S. (2023). TrackNetV3: Enhancing ShuttleCock
Tracking with Augmentations and Trajectory Rectification.

## Ultralytics YOLO (person detection) — AGPL-3.0

Player detection uses the `ultralytics` package and the YOLO11n weights, both
licensed under **AGPL-3.0** (https://www.ultralytics.com/license). AGPL
obligations extend to software offered over a network. Before any commercial
or closed-source deployment, either obtain an Ultralytics Enterprise licence or
replace `YoloPersonDetector` (behind the `PersonDetector` interface in
`backend/rallycv/players.py`) with a permissively licensed detector.

## Sample footage (not distributed)

`backend/data/samples/` (git-ignored) holds clips used for manual testing,
downloaded from Wikimedia Commons under CC BY 3.0:
"Wang Zhiyi triumph in all-Chinese China Open finals I China Open 2025".
