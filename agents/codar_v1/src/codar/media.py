from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional


class MediaResolver:
    def __init__(self, runtime_cfg: Dict[str, Any]):
        self.pipeline_cfg = runtime_cfg.get("pipeline", {})
        self.media_cfg = runtime_cfg.get("media", {})
        raw_mode = self.pipeline_cfg.get("media_mode", "off")
        if isinstance(raw_mode, bool):
            mode = "off" if raw_mode is False else "local"
        else:
            mode = str(raw_mode).strip().lower()
        self.media_mode = mode if mode in {"off", "local", "auto"} else "off"
        self.max_video_frames = int(self.pipeline_cfg.get("max_video_frames", 4))
        self.remote_root = Path(str(self.media_cfg.get("remote_root", "")))
        self.allow_url_fallback = bool(self.media_cfg.get("allow_url_fallback", True))

    def resolve(self, sample_id: str, scenario: str, media: Dict[str, Any]) -> Dict[str, Any]:
        if self.media_mode == "off":
            return {
                "mode": "off",
                "image": None,
                "video": None,
                "audio": None,
                "frames": [],
                "image_url": media.get("image_url"),
                "video_url": media.get("video_url"),
                "audio_caption": media.get("audio_caption", ""),
            }

        manifest = {
            "mode": self.media_mode,
            "image": self._resolve_image(sample_id, scenario, media),
            "video": self._resolve_video(sample_id, scenario, media),
            "audio": self._resolve_audio(sample_id, scenario, media),
            "frames": self._resolve_frames(sample_id, scenario, media),
            "image_url": media.get("image_url"),
            "video_url": media.get("video_url"),
            "audio_caption": media.get("audio_caption", ""),
        }
        return manifest

    def _resolve_image(self, sample_id: str, scenario: str, media: Dict[str, Any]) -> Optional[str]:
        base = self.remote_root / scenario / "image"
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = base / f"{sample_id}{ext}"
            if p.exists():
                return str(p)
        for key in ("image_path",):
            v = str(media.get(key, "")).strip()
            if v and Path(v).exists():
                return v
        if self.allow_url_fallback:
            return str(media.get("image_url", "")).strip() or None
        return None

    def _resolve_video(self, sample_id: str, scenario: str, media: Dict[str, Any]) -> Optional[str]:
        p = self.remote_root / scenario / "video" / f"{sample_id}.mp4"
        if p.exists():
            return str(p)
        v = str(media.get("video_path", "")).strip()
        if v and Path(v).exists():
            return v
        if self.allow_url_fallback:
            return str(media.get("video_url", "")).strip() or None
        return None

    def _resolve_audio(self, sample_id: str, scenario: str, media: Dict[str, Any]) -> Optional[str]:
        p = self.remote_root / scenario / "Video_composition" / "audio_caption" / f"{sample_id}.mp3"
        if p.exists():
            return str(p)
        v = str(media.get("audio_path", "")).strip()
        if v and Path(v).exists():
            return v
        if self.allow_url_fallback:
            return str(media.get("audio_url", "")).strip() or None
        return None

    def _resolve_frames(self, sample_id: str, scenario: str, media: Dict[str, Any]) -> List[str]:
        frame_dir = self.remote_root / scenario / "Video_composition" / "frame" / sample_id
        frames: List[str] = []
        if frame_dir.exists():
            for p in sorted(frame_dir.glob("*.jpg"))[: self.max_video_frames]:
                frames.append(str(p))
            if frames:
                return frames
        if self.allow_url_fallback:
            url_prefix = str(media.get("url_frame", "")).strip()
            frame_count = int(media.get("frame_count", 0) or 0)
            if url_prefix and frame_count > 0:
                for i in range(1, min(frame_count, self.max_video_frames) + 1):
                    frames.append(f"{url_prefix}/{sample_id}_frame{i}.jpg")
        return frames

    @staticmethod
    def to_data_url(path: str) -> str:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return path
        suffix = p.suffix.lower()
        mime = "image/jpeg"
        if suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
