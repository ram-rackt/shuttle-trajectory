"""Local storage: one SQLite database plus content-addressed files.

Layout (under Settings.storage_dir, git-ignored):

    rallyreview.sqlite3     jobs, videos, downloads, cached results
    videos/<sha256>.<ext>   each distinct video stored once, named by its content
    cache/<sha256>/         per-video pipeline stage cache (see rallycv.cache)
    tmp/                    uploads and downloads in progress

A video is identified by the SHA-256 of its bytes, so the same file uploaded
twice (under any name) is stored once and analysed once: a repeat analysis
with the same settings is answered from the results table.

For a hosted deployment, the same interface can be backed by Postgres and
object storage; nothing outside this module touches the database or paths.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,  -- sha256 of the file
    ext           TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    name          TEXT NOT NULL,     -- original file name or video title
    source        TEXT NOT NULL,     -- 'upload' | 'youtube'
    youtube_id    TEXT,
    channel       TEXT,
    created_at    REAL NOT NULL,
    last_used_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS downloads (
    source_key  TEXT PRIMARY KEY,    -- e.g. 'youtube:<id>:<format>'
    video_id    TEXT NOT NULL REFERENCES videos(id),
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    key         TEXT PRIMARY KEY,    -- video + analyser version + request settings
    video_id    TEXT NOT NULL REFERENCES videos(id),
    json        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,     -- 'upload' | 'youtube'
    video_id      TEXT REFERENCES videos(id),  -- null until a YouTube video is downloaded
    status        TEXT NOT NULL,
    progress      REAL NOT NULL,
    current_stage INTEGER,
    stages        TEXT NOT NULL,     -- JSON list of stage labels
    error         TEXT,
    error_code    TEXT,
    request       TEXT NOT NULL,     -- JSON: settings the analysis was asked for
    result_key    TEXT,
    cached        INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_video ON jobs(video_id);
"""


@dataclass(frozen=True)
class VideoRecord:
    id: str
    ext: str
    size_bytes: int
    name: str
    source: str
    youtube_id: str | None = None
    channel: str | None = None


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.videos_dir = root / "videos"
        self.cache_root = root / "cache"
        self.tmp_dir = root / "tmp"
        for d in (self.videos_dir, self.cache_root, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._clean_tmp()
        self._lock = threading.RLock()
        # One connection shared by the request handlers and job threads, guarded
        # by a lock: SQLite serialises writes anyway, and this keeps it simple.
        self._db = sqlite3.connect(
            root / "rallyreview.sqlite3", check_same_thread=False, isolation_level=None
        )
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _clean_tmp(self) -> None:
        """Leftovers from a crash or restart mid-upload/download."""
        for p in self.tmp_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)

    # ---- Files ---------------------------------------------------------------

    def new_tmp_path(self, suffix: str = "") -> Path:
        return self.tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def add_video(
        self,
        tmp_path: Path,
        sha256: str,
        ext: str,
        *,
        name: str,
        source: str,
        youtube_id: str | None = None,
        channel: str | None = None,
    ) -> VideoRecord:
        """Moves a received file into the store, or discards it if the same
        content is already stored. Returns the stored video."""
        existing = self.video(sha256)
        if existing is not None and self.video_path(existing).exists():
            tmp_path.unlink(missing_ok=True)
            self.touch_video(sha256)
            return existing
        dest = self.videos_dir / f"{sha256}{ext}"
        os.replace(tmp_path, dest)
        now = time.time()
        record = VideoRecord(sha256, ext, dest.stat().st_size, name, source, youtube_id, channel)
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?,?,?)",
                (sha256, ext, record.size_bytes, name, source, youtube_id, channel, now, now),
            )
        return record

    def video(self, video_id: str) -> VideoRecord | None:
        rows = self._query(
            "SELECT id, ext, size_bytes, name, source, youtube_id, channel FROM videos WHERE id=?",
            (video_id,),
        )
        return VideoRecord(**dict(rows[0])) if rows else None

    def video_path(self, video: VideoRecord) -> Path:
        return self.videos_dir / f"{video.id}{video.ext}"

    def touch_video(self, video_id: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE videos SET last_used_at=? WHERE id=?", (time.time(), video_id))

    def cache_dir(self, video_id: str) -> Path:
        return self.cache_root / video_id

    # ---- Downloads -----------------------------------------------------------

    def downloaded(self, source_key: str) -> VideoRecord | None:
        rows = self._query("SELECT video_id FROM downloads WHERE source_key=?", (source_key,))
        if not rows:
            return None
        video = self.video(rows[0]["video_id"])
        return video if video is not None and self.video_path(video).exists() else None

    def record_download(self, source_key: str, video_id: str) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO downloads VALUES (?,?,?)",
                (source_key, video_id, time.time()),
            )

    # ---- Results -------------------------------------------------------------

    @staticmethod
    def result_key(video_id: str, analyzer_version: str, request: dict) -> str:
        blob = json.dumps(
            {"video": video_id, "analyzer": analyzer_version, "request": request}, sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def result(self, key: str) -> str | None:
        rows = self._query("SELECT json FROM results WHERE key=?", (key,))
        return rows[0]["json"] if rows else None

    def put_result(self, key: str, video_id: str, result_json: str) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                (key, video_id, result_json, time.time()),
            )

    # ---- Jobs ----------------------------------------------------------------

    def insert_job(self, row: dict) -> None:
        now = time.time()
        row = {**row, "created_at": now, "updated_at": now}
        cols = ",".join(row)
        with self._tx() as db:
            db.execute(
                f"INSERT INTO jobs ({cols}) VALUES ({','.join('?' * len(row))})",
                tuple(row.values()),
            )

    def update_job(self, job_id: str, **changes: object) -> None:
        if not changes:
            return
        changes["updated_at"] = time.time()
        sets = ",".join(f"{k}=?" for k in changes)
        with self._tx() as db:
            db.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*changes.values(), job_id))

    def job_row(self, job_id: str) -> dict | None:
        rows = self._query("SELECT * FROM jobs WHERE id=?", (job_id,))
        return dict(rows[0]) if rows else None

    def fail_interrupted_jobs(self) -> int:
        """Jobs still marked as running belonged to a previous server process."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE jobs SET status='failed', current_stage=NULL, error_code='interrupted', "
                "error='The server restarted while this was running. Please try again.', "
                "updated_at=? WHERE status IN ('queued', 'processing')",
                (time.time(),),
            )
            return cur.rowcount
