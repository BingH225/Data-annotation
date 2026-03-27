from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..backends.base import BaseBackend
from ..media import MediaResolver
from ..prompting import PromptStore
from ..types import PromptMeta, StageArtifact
from ..utils import clip_text


def build_media_items(media_manifest: Dict[str, Any], max_frames: int = 4) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    image = media_manifest.get("image")
    if isinstance(image, str) and image:
        url = image
        if not (image.startswith("http://") or image.startswith("https://") or image.startswith("data:")):
            url = MediaResolver.to_data_url(image)
        items.append({"kind": "image", "url": url})
    frames = media_manifest.get("frames", []) or []
    for frame in frames[:max_frames]:
        if isinstance(frame, str) and frame:
            url = frame
            if not (frame.startswith("http://") or frame.startswith("https://") or frame.startswith("data:")):
                url = MediaResolver.to_data_url(frame)
            items.append({"kind": "frame", "url": url})
    return items


def run_prompt_json(
    backend: BaseBackend,
    prompt_store: PromptStore,
    stage_id: str,
    prompt_id: str,
    prompt_vars: Dict[str, Any],
    max_retries: int,
    media_items: Optional[List[Dict[str, Any]]] = None,
    temperature_override: Optional[float] = None,
) -> Tuple[Dict[str, Any], StageArtifact]:
    retries = 0
    last_err = ""
    while retries <= max_retries:
        try:
            rendered = prompt_store.render(prompt_id=prompt_id, prompt_vars=prompt_vars)
            rsp = backend.complete_json(
                prompt_text=rendered.text,
                prompt_id=prompt_id,
                media_items=media_items,
                temperature_override=temperature_override,
            )
            artifact = StageArtifact(
                stage_id=stage_id,
                status="ok",
                output=rsp.parsed_json,
                prompt_meta=PromptMeta(
                    prompt_id=rendered.meta.prompt_id,
                    prompt_vars=rendered.meta.prompt_vars,
                    prompt_hash=rendered.meta.prompt_hash,
                ),
                retries=retries,
            )
            return rsp.parsed_json, artifact
        except Exception as exc:  # pragma: no cover - exercised via integration paths
            last_err = str(exc)
            retries += 1
    artifact = StageArtifact(
        stage_id=stage_id,
        status="error",
        output={},
        prompt_meta=None,
        retries=retries,
        notes=clip_text(last_err, 600),
    )
    return {}, artifact
