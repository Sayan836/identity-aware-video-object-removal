from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class JobRecord:
    """JSON-friendly state persisted for one processing job."""

    id: str
    input_path: str
    output_path: str
    state: str = "queued"
    phase: str = "queued"
    current_frame: int = 0
    total_frames: int | None = None
    error: str | None = None
    download_url: str | None = None
    preview_mask_url: str | None = None
    logs: list[dict[str, str]] | None = None

    def snapshot(self) -> dict[str, object]:
        """Return the API response shape used by the browser UI."""

        return {
            "id": self.id,
            "state": self.state,
            "phase": self.phase,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "error": self.error,
            "download_url": self.download_url,
            "preview_mask_url": self.preview_mask_url,
            "logs": self.logs or [],
        }


class RedisJobStore:
    """Store job progress in Redis so web and worker processes share state."""

    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None) -> None:
        self.redis_url = redis_url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
        raw_ttl = os.environ.get("JOB_STATE_TTL_SECONDS")
        self.ttl_seconds = ttl_seconds or int(raw_ttl or DEFAULT_JOB_TTL_SECONDS)
        self._client = None

    @property
    def client(self):
        """Create the Redis client lazily so imports do not require Redis."""

        if self._client is None:
            try:
                import redis  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "Redis support requires the 'redis' Python package. "
                    "Install dependencies with: python3 -m pip install -r requirements.txt"
                ) from exc
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def ping(self) -> None:
        """Validate that Redis is reachable."""

        self.client.ping()

    def create(self, record: JobRecord) -> None:
        """Persist a new job record."""

        key = self._key(record.id)
        self.client.hset(key, mapping=self._encode(asdict(record)))
        self.client.expire(key, self.ttl_seconds)

    def get(self, job_id: str) -> JobRecord | None:
        """Fetch one job record, returning None if it has expired or never existed."""

        raw = self.client.hgetall(self._key(job_id))
        if not raw:
            return None
        return JobRecord(
            id=raw["id"],
            input_path=raw["input_path"],
            output_path=raw["output_path"],
            state=raw.get("state", "queued"),
            phase=raw.get("phase", "queued"),
            current_frame=int(raw.get("current_frame") or 0),
            total_frames=self._optional_int(raw.get("total_frames")),
            error=self._optional_str(raw.get("error")),
            download_url=self._optional_str(raw.get("download_url")),
            preview_mask_url=self._optional_str(raw.get("preview_mask_url")),
            logs=self._decode_logs(raw.get("logs")),
        )

    def update(self, job_id: str, **fields: Any) -> None:
        """Patch selected job fields and refresh the expiry window."""

        key = self._key(job_id)
        self.client.hset(key, mapping=self._encode(fields))
        self.client.expire(key, self.ttl_seconds)

    def update_progress(self, job_id: str, current_frame: int, total_frames: int | None) -> None:
        """Store frame progress emitted by the inpainting pipeline."""

        self.update(job_id, current_frame=current_frame, total_frames=total_frames)

    def update_phase(self, job_id: str, phase: str) -> None:
        """Store the current pipeline phase for UI polling."""

        self.update(job_id, phase=phase)

    def append_log(self, job_id: str, message: str, phase: str | None = None) -> None:
        """Append one compact job log entry for the browser UI."""

        record = self.get(job_id)
        logs = list(record.logs or []) if record else []
        logs.append(
            {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "phase": phase or (record.phase if record else ""),
                "message": message,
            }
        )
        self.update(job_id, logs=logs[-300:])

    def output_path(self, job_id: str) -> Path | None:
        """Return the completed output path for a known job."""

        record = self.get(job_id)
        return Path(record.output_path) if record else None

    def _key(self, job_id: str) -> str:
        return f"video-object-removal:job:{job_id}"

    def _encode(self, fields: dict[str, Any]) -> dict[str, str]:
        encoded: dict[str, str] = {}
        for key, value in fields.items():
            if key == "logs" and value is not None:
                encoded[key] = json.dumps(value)
            else:
                encoded[key] = "" if value is None else str(value)
        return encoded

    def _optional_int(self, value: str | None) -> int | None:
        return int(value) if value else None

    def _optional_str(self, value: str | None) -> str | None:
        return value or None

    def _decode_logs(self, value: str | None) -> list[dict[str, str]]:
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if not isinstance(decoded, list):
            return []
        logs: list[dict[str, str]] = []
        for item in decoded:
            if not isinstance(item, dict):
                continue
            logs.append(
                {
                    "time": str(item.get("time") or ""),
                    "phase": str(item.get("phase") or ""),
                    "message": str(item.get("message") or ""),
                }
            )
        return logs
