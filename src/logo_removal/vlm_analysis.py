from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .mask_providers import (
    AI_HUMAN_CROWDED,
    AI_OBJECT,
    AI_TEXT_OR_LOGO,
    MaskProvider,
    STATIC_RECTANGLE,
)
from .video import VideoMetadata


VLM_PROVIDER_NONE = "none"
VLM_PROVIDER_HEURISTIC = "heuristic"
VLM_PROVIDER_OLLAMA = "ollama"
VALID_VLM_PROVIDERS = {
    VLM_PROVIDER_NONE,
    VLM_PROVIDER_HEURISTIC,
    VLM_PROVIDER_OLLAMA,
}

DEFAULT_HEURISTIC_CONTACT_DILATION_PX = 10
DEFAULT_HEURISTIC_SHADOW_DILATION_PX = 30
DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX = 16
DEFAULT_HEURISTIC_SHADOW_HORIZONTAL_OFFSET_PX = 0
DEFAULT_OLLAMA_VLM_MODEL = "llama3.2-vision:11b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 180
DEFAULT_OLLAMA_SAMPLE_FRAMES = 3
DEFAULT_OLLAMA_MAX_IMAGE_SIDE = 768

_DEFAULT_VOID_PROMPT = "clean natural background after the selected object is removed"


@dataclass(frozen=True)
class PromptGenerationResult:
    """Generated prompt text plus compact metadata for manifests."""

    prompt: str
    strategy: str
    vlm_provider: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VlmAnalysisResult:
    """Compact scene reasoning used to condition VOID prompt and affected masks."""

    provider: str
    model: str | None = None
    scene_description: str = ""
    prompt_hint: str = ""
    affected_region_notes: tuple[str, ...] = ()
    contact_dilation_px: int | None = None
    shadow_dilation_px: int | None = None
    shadow_vertical_offset_px: int | None = None
    shadow_horizontal_offset_px: int | None = None
    confidence: float = 0.0
    raw_response: str | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.confidence >= 0.2

    def to_manifest(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "scene_description": self.scene_description,
            "prompt_hint": self.prompt_hint,
            "affected_region_notes": list(self.affected_region_notes),
            "contact_dilation_px": self.contact_dilation_px,
            "shadow_dilation_px": self.shadow_dilation_px,
            "shadow_vertical_offset_px": self.shadow_vertical_offset_px,
            "shadow_horizontal_offset_px": self.shadow_horizontal_offset_px,
            "confidence": self.confidence,
            "error": self.error,
        }


@dataclass
class HeuristicAffectedMaskProvider:
    """Build VOID affected-region masks from primary-mask geometry only."""

    np: object
    cv2: object
    metadata: VideoMetadata
    primary_mask_provider: MaskProvider
    removal_mode: str = AI_OBJECT
    contact_dilation_px: int = DEFAULT_HEURISTIC_CONTACT_DILATION_PX
    shadow_dilation_px: int = DEFAULT_HEURISTIC_SHADOW_DILATION_PX
    shadow_vertical_offset_px: int = DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX
    shadow_horizontal_offset_px: int = DEFAULT_HEURISTIC_SHADOW_HORIZONTAL_OFFSET_PX

    provider_name: str = VLM_PROVIDER_HEURISTIC

    def __post_init__(self) -> None:
        _ensure_non_negative("contact_dilation_px", self.contact_dilation_px)
        _ensure_non_negative("shadow_dilation_px", self.shadow_dilation_px)
        _ensure_non_negative("shadow_vertical_offset_px", self.shadow_vertical_offset_px)

    @property
    def config(self) -> dict[str, object]:
        """Return serializable settings for package manifests."""

        return {
            "provider": self.provider_name,
            "removal_mode": self.removal_mode,
            "contact_dilation_px": self.contact_dilation_px,
            "shadow_dilation_px": self.shadow_dilation_px,
            "shadow_vertical_offset_px": self.shadow_vertical_offset_px,
            "shadow_horizontal_offset_px": self.shadow_horizontal_offset_px,
            "notes": [
                "Heuristic only; no external VLM call.",
                "Primary mask is never expanded into neighboring people by this provider.",
            ],
        }

    def prepare(self) -> None:
        """The primary provider is prepared by the caller/exporter."""

    def mask_for_frame(self, frame_index: int, frame):
        """Return the affected-only mask for a frame."""

        primary_mask = self.primary_mask_provider.mask_for_frame(frame_index, frame)
        return self.mask_from_primary_mask(primary_mask, frame_index=frame_index, frame=frame)

    def mask_from_primary_mask(self, primary_mask, frame_index: int | None = None, frame=None):
        """Create affected regions around the already-computed primary mask."""

        del frame_index
        primary_mask = self._resize_to_target(primary_mask, frame)
        primary = primary_mask > 0
        if int(self.np.count_nonzero(primary)) == 0:
            return self.np.zeros(primary_mask.shape[:2], dtype=self.np.uint8)

        contact = self._contact_region(primary)
        shadow = self._shadow_region(primary)
        affected = (contact | shadow) & ~primary
        return (affected.astype(self.np.uint8) * 255)

    def _resize_to_target(self, mask, frame):
        target_height = self.metadata.height
        target_width = self.metadata.width
        if frame is not None:
            target_height, target_width = frame.shape[:2]
        if mask.shape[:2] == (target_height, target_width):
            return mask
        return self.cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=self.cv2.INTER_NEAREST,
        )

    def _contact_region(self, primary):
        if self.contact_dilation_px <= 0:
            return self.np.zeros(primary.shape, dtype=bool)
        dilated = self._dilate(primary, self.contact_dilation_px)
        return dilated & ~primary

    def _shadow_region(self, primary):
        if (
            self.shadow_dilation_px <= 0
            and self.shadow_vertical_offset_px == 0
            and self.shadow_horizontal_offset_px == 0
        ):
            return self.np.zeros(primary.shape, dtype=bool)

        shifted = self._shift_mask(
            primary.astype(self.np.uint8),
            x_offset=self.shadow_horizontal_offset_px,
            y_offset=self.shadow_vertical_offset_px,
        )

        if self.shadow_dilation_px > 0:
            shadow = self._dilate(shifted > 0, self.shadow_dilation_px)
        else:
            shadow = shifted > 0

        primary_rows = self.np.where(primary)[0]
        if primary_rows.size:
            shadow[: int(primary_rows.min()), :] = False
        return shadow & ~primary

    def _shift_mask(self, mask, x_offset: int, y_offset: int):
        height, width = mask.shape[:2]
        x_offset = max(-width + 1, min(int(x_offset), width - 1)) if width else 0
        y_offset = max(-height + 1, min(int(y_offset), height - 1)) if height else 0

        shifted = self.np.zeros(mask.shape[:2], dtype=mask.dtype)
        source_x0 = max(0, -x_offset)
        source_x1 = width - max(0, x_offset)
        dest_x0 = max(0, x_offset)
        dest_x1 = width - max(0, -x_offset)
        source_y0 = max(0, -y_offset)
        source_y1 = height - max(0, y_offset)
        dest_y0 = max(0, y_offset)
        dest_y1 = height - max(0, -y_offset)
        if source_x0 < source_x1 and source_y0 < source_y1:
            shifted[dest_y0:dest_y1, dest_x0:dest_x1] = mask[
                source_y0:source_y1,
                source_x0:source_x1,
            ]
        return shifted

    def _dilate(self, mask, radius_px: int):
        kernel_size = radius_px * 2 + 1
        kernel = self.cv2.getStructuringElement(
            self.cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return self.cv2.dilate(mask.astype(self.np.uint8), kernel, iterations=1) > 0


@dataclass
class OllamaAffectedMaskProvider(HeuristicAffectedMaskProvider):
    """Use Ollama VLM scene analysis to tune deterministic affected masks."""

    input_path: Path | None = None
    vlm_analysis: VlmAnalysisResult | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    ollama_timeout_seconds: float | None = None
    ollama_sample_frames: int | None = None
    ollama_max_image_side: int | None = None

    provider_name: str = VLM_PROVIDER_OLLAMA

    @property
    def analysis(self) -> VlmAnalysisResult | None:
        return self.vlm_analysis

    @property
    def config(self) -> dict[str, object]:
        config = super().config
        config.update(
            {
                "provider": self.provider_name,
                "fallback_provider": VLM_PROVIDER_HEURISTIC,
                "ollama_model": self.ollama_model
                or os.environ.get("OLLAMA_VLM_MODEL")
                or DEFAULT_OLLAMA_VLM_MODEL,
                "ollama_base_url": _resolve_ollama_base_url(self.ollama_base_url),
                "analysis": self.vlm_analysis.to_manifest()
                if self.vlm_analysis is not None
                else None,
                "notes": [
                    "Ollama VLM tunes deterministic contact/shadow/reflection heuristics.",
                    "Falls back to heuristic settings when Ollama is unavailable or low confidence.",
                    "Primary target mask remains excluded from affected-only output.",
                ],
            }
        )
        return config

    def prepare(self) -> None:
        if self.vlm_analysis is None:
            self.vlm_analysis = analyze_ollama_scene(
                np=self.np,
                cv2=self.cv2,
                metadata=self.metadata,
                primary_mask_provider=self.primary_mask_provider,
                removal_mode=self.removal_mode,
                input_path=self.input_path,
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                timeout_seconds=self.ollama_timeout_seconds,
                sample_frame_count=self.ollama_sample_frames,
                max_image_side=self.ollama_max_image_side,
            )
        if self.vlm_analysis.usable:
            self._apply_vlm_analysis(self.vlm_analysis)

    def _apply_vlm_analysis(self, analysis: VlmAnalysisResult) -> None:
        self.contact_dilation_px = _bounded_int(
            analysis.contact_dilation_px,
            default=self.contact_dilation_px,
            minimum=0,
            maximum=48,
        )
        self.shadow_dilation_px = _bounded_int(
            analysis.shadow_dilation_px,
            default=self.shadow_dilation_px,
            minimum=0,
            maximum=96,
        )
        self.shadow_vertical_offset_px = _bounded_int(
            analysis.shadow_vertical_offset_px,
            default=self.shadow_vertical_offset_px,
            minimum=0,
            maximum=72,
        )
        self.shadow_horizontal_offset_px = _bounded_int(
            analysis.shadow_horizontal_offset_px,
            default=self.shadow_horizontal_offset_px,
            minimum=-72,
            maximum=72,
        )


def validate_vlm_provider(vlm_provider: str) -> None:
    """Validate the currently supported affected-region provider mode."""

    if vlm_provider not in VALID_VLM_PROVIDERS:
        raise ValueError(
            "vlm_provider must be one of: "
            f"{', '.join(sorted(VALID_VLM_PROVIDERS))}."
        )


def build_affected_mask_provider(
    *,
    vlm_provider: str,
    np,
    cv2,
    metadata: VideoMetadata,
    primary_mask_provider: MaskProvider,
    removal_mode: str,
    input_path: Path | None = None,
    vlm_analysis: VlmAnalysisResult | None = None,
    contact_dilation_px: int = DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    shadow_dilation_px: int = DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    shadow_vertical_offset_px: int = DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    shadow_horizontal_offset_px: int = DEFAULT_HEURISTIC_SHADOW_HORIZONTAL_OFFSET_PX,
) -> MaskProvider | None:
    """Build the affected-region provider requested by the VOID config."""

    validate_vlm_provider(vlm_provider)
    if vlm_provider == VLM_PROVIDER_NONE:
        return None
    provider_cls = (
        OllamaAffectedMaskProvider
        if vlm_provider == VLM_PROVIDER_OLLAMA
        else HeuristicAffectedMaskProvider
    )
    kwargs = {}
    if provider_cls is OllamaAffectedMaskProvider:
        kwargs.update(
            {
                "input_path": input_path,
                "vlm_analysis": vlm_analysis,
            }
        )
    return provider_cls(
        np=np,
        cv2=cv2,
        metadata=metadata,
        primary_mask_provider=primary_mask_provider,
        removal_mode=removal_mode,
        contact_dilation_px=contact_dilation_px,
        shadow_dilation_px=shadow_dilation_px,
        shadow_vertical_offset_px=shadow_vertical_offset_px,
        shadow_horizontal_offset_px=shadow_horizontal_offset_px,
        **kwargs,
    )


def generate_void_prompt(
    *,
    base_prompt: str,
    removal_mode: str,
    vlm_provider: str,
    vlm_analysis: VlmAnalysisResult | None = None,
) -> PromptGenerationResult:
    """Generate conservative VOID prompt text for the selected mode."""

    validate_vlm_provider(vlm_provider)
    base_prompt = (base_prompt or _DEFAULT_VOID_PROMPT).strip() or _DEFAULT_VOID_PROMPT
    custom_base = base_prompt != _DEFAULT_VOID_PROMPT
    affected_phrase = ""
    if vlm_provider == VLM_PROVIDER_HEURISTIC:
        affected_phrase = " and nearby contact, shadow, reflection, or edge artifacts"
    elif vlm_provider == VLM_PROVIDER_OLLAMA:
        affected_phrase = (
            " and VLM-identified contact, shadow, reflection, occlusion-edge, "
            "or nearby interaction artifacts"
        )
    prompt_hint = _analysis_prompt_hint(vlm_analysis)
    strategy_suffix = "_ollama_scene" if vlm_provider == VLM_PROVIDER_OLLAMA else ""

    if removal_mode == STATIC_RECTANGLE:
        return PromptGenerationResult(
            prompt=_join_sentences([base_prompt, prompt_hint]) if prompt_hint else base_prompt,
            strategy=f"manual_static{strategy_suffix}",
            vlm_provider=vlm_provider,
            notes=_prompt_notes(
                base_notes=("Static rectangle keeps the supplied prompt unchanged.",),
                vlm_analysis=vlm_analysis,
            ),
        )

    if removal_mode == AI_HUMAN_CROWDED:
        mode_prompt = (
            f"Remove only the selected target person{affected_phrase}. Preserve every "
            "other person, limb, clothing item, object, floor texture, lighting, "
            "camera motion, and background layout. Reconstruct the occluded "
            "background naturally without duplicate bodies, ghost limbs, or changes "
            "to neighboring people."
        )
        mode_prompt = _join_sentences([mode_prompt, prompt_hint])
        return _prompt_result(
            base_prompt=base_prompt,
            mode_prompt=mode_prompt,
            custom_base=custom_base,
            strategy=f"crowded_human_preservation{strategy_suffix}",
            vlm_provider=vlm_provider,
            vlm_analysis=vlm_analysis,
        )

    if removal_mode == AI_TEXT_OR_LOGO:
        mode_prompt = (
            f"Remove only the selected text or logo{affected_phrase}. Preserve "
            "the surrounding material, edges, lighting, texture, camera motion, "
            "and background layout without introducing replacement text."
        )
        mode_prompt = _join_sentences([mode_prompt, prompt_hint])
        return _prompt_result(
            base_prompt=base_prompt,
            mode_prompt=mode_prompt,
            custom_base=custom_base,
            strategy=f"text_logo_cleanup{strategy_suffix}",
            vlm_provider=vlm_provider,
            vlm_analysis=vlm_analysis,
        )

    mode_prompt = (
        f"Remove only the selected target object{affected_phrase}. Preserve "
        "surrounding people, objects, surfaces, lighting, texture, camera motion, "
        "and background layout. Reconstruct the occluded background naturally."
    )
    mode_prompt = _join_sentences([mode_prompt, prompt_hint])
    return _prompt_result(
        base_prompt=base_prompt,
        mode_prompt=mode_prompt,
        custom_base=custom_base,
        strategy=f"object_preservation{strategy_suffix}",
        vlm_provider=vlm_provider,
        vlm_analysis=vlm_analysis,
    )


def analyze_ollama_scene(
    *,
    np,
    cv2,
    metadata: VideoMetadata,
    primary_mask_provider: MaskProvider | None,
    removal_mode: str,
    input_path: Path | None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    sample_frame_count: int | None = None,
    max_image_side: int | None = None,
) -> VlmAnalysisResult:
    """Call a local Ollama vision model on sampled overlay frames."""

    model_name = model or os.environ.get("OLLAMA_VLM_MODEL") or DEFAULT_OLLAMA_VLM_MODEL
    resolved_base_url = _resolve_ollama_base_url(base_url)
    timeout = float(
        timeout_seconds
        or os.environ.get("OLLAMA_VLM_TIMEOUT_SECONDS")
        or DEFAULT_OLLAMA_TIMEOUT_SECONDS
    )
    sample_count = int(
        sample_frame_count
        or os.environ.get("OLLAMA_VLM_SAMPLE_FRAMES")
        or DEFAULT_OLLAMA_SAMPLE_FRAMES
    )
    image_side = int(
        max_image_side
        or os.environ.get("OLLAMA_VLM_MAX_IMAGE_SIDE")
        or DEFAULT_OLLAMA_MAX_IMAGE_SIDE
    )
    try:
        images = _sample_overlay_images(
            np=np,
            cv2=cv2,
            metadata=metadata,
            primary_mask_provider=primary_mask_provider,
            input_path=input_path,
            sample_frame_count=sample_count,
            max_image_side=image_side,
        )
        if not images:
            return VlmAnalysisResult(
                provider=VLM_PROVIDER_OLLAMA,
                model=model_name,
                error="No usable sampled frames were available for Ollama VLM analysis.",
            )
        response_text = _call_ollama_generate(
            base_url=resolved_base_url,
            model=model_name,
            prompt=_ollama_analysis_prompt(removal_mode),
            images=images,
            timeout_seconds=timeout,
        )
        return _parse_ollama_analysis_response(response_text, model_name)
    except Exception as exc:
        return VlmAnalysisResult(
            provider=VLM_PROVIDER_OLLAMA,
            model=model_name,
            error=str(exc),
        )


def _prompt_result(
    *,
    base_prompt: str,
    mode_prompt: str,
    custom_base: bool,
    strategy: str,
    vlm_provider: str,
    vlm_analysis: VlmAnalysisResult | None = None,
) -> PromptGenerationResult:
    if custom_base:
        prompt = _join_sentences([base_prompt, mode_prompt])
        notes = _prompt_notes(
            base_notes=("Supplied prompt preserved and augmented with mode-aware constraints.",),
            vlm_analysis=vlm_analysis,
        )
    else:
        prompt = mode_prompt
        notes = _prompt_notes(
            base_notes=("Default prompt replaced with mode-aware constraints.",),
            vlm_analysis=vlm_analysis,
        )
    return PromptGenerationResult(
        prompt=prompt,
        strategy=strategy,
        vlm_provider=vlm_provider,
        notes=notes,
    )


def _join_sentences(parts: Iterable[str]) -> str:
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part[-1] not in ".!?":
            part = f"{part}."
        sentences.append(part)
    return " ".join(sentences)


def _ensure_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _prompt_notes(
    *,
    base_notes: tuple[str, ...],
    vlm_analysis: VlmAnalysisResult | None,
) -> tuple[str, ...]:
    if vlm_analysis is None:
        return base_notes
    notes = list(base_notes)
    if vlm_analysis.error:
        notes.append(f"Ollama VLM unavailable; heuristic fallback used: {vlm_analysis.error}")
    elif vlm_analysis.prompt_hint:
        notes.append("Ollama VLM scene guidance added to prompt.")
    if vlm_analysis.affected_region_notes:
        notes.extend(vlm_analysis.affected_region_notes[:3])
    return tuple(notes)


def _analysis_prompt_hint(vlm_analysis: VlmAnalysisResult | None) -> str:
    if vlm_analysis is None or not vlm_analysis.usable:
        return ""
    hint = _clean_text(vlm_analysis.prompt_hint)
    if not hint:
        description = _clean_text(vlm_analysis.scene_description)
        if description:
            hint = f"Reconstruct the removed area as consistent with this scene: {description}"
    if not hint:
        return ""
    return f"Scene guidance from local Ollama VLM: {hint}"


def _sample_overlay_images(
    *,
    np,
    cv2,
    metadata: VideoMetadata,
    primary_mask_provider: MaskProvider | None,
    input_path: Path | None,
    sample_frame_count: int,
    max_image_side: int,
) -> list[str]:
    if input_path is None or primary_mask_provider is None:
        return []
    input_path = Path(input_path)
    if not input_path.exists():
        return []

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return []

    encoded_images: list[str] = []
    frame_count = int(metadata.frame_count or capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidates = _sample_one_based_indices(frame_count, sample_frame_count)
    try:
        for frame_index in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index - 1, 0))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            mask = primary_mask_provider.mask_for_frame(frame_index, frame)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if int(np.count_nonzero(mask > 0)) == 0:
                continue
            overlay = _build_overlay_frame(np=np, cv2=cv2, frame=frame, mask=mask)
            overlay = _resize_for_vlm(cv2=cv2, frame=overlay, max_image_side=max_image_side)
            ok, buffer = cv2.imencode(
                ".jpg",
                overlay,
                [int(cv2.IMWRITE_JPEG_QUALITY), 88],
            )
            if ok:
                encoded_images.append(base64.b64encode(buffer.tobytes()).decode("ascii"))
    finally:
        capture.release()
    return encoded_images


def _build_overlay_frame(*, np, cv2, frame, mask):
    overlay = frame.copy()
    primary = mask > 0
    red = np.zeros_like(frame)
    red[:, :, 2] = 255
    overlay[primary] = (0.55 * frame[primary] + 0.45 * red[primary]).astype(frame.dtype)
    contours, _ = cv2.findContours(
        primary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def _resize_for_vlm(*, cv2, frame, max_image_side: int):
    max_image_side = max(256, int(max_image_side))
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_image_side:
        return frame
    scale = max_image_side / float(longest)
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _sample_one_based_indices(frame_count: int, sample_frame_count: int) -> list[int]:
    frame_count = max(1, int(frame_count or 1))
    sample_frame_count = max(1, int(sample_frame_count or 1))
    if sample_frame_count == 1 or frame_count == 1:
        return [max(1, int(round(frame_count / 2)))]
    indices = [
        int(round(1 + index * (frame_count - 1) / (sample_frame_count - 1)))
        for index in range(sample_frame_count)
    ]
    return sorted({max(1, min(frame_count, index)) for index in indices})


def _ollama_analysis_prompt(removal_mode: str) -> str:
    target_label = "selected target person" if removal_mode == AI_HUMAN_CROWDED else "selected target object"
    return (
        "You are helping a video object-removal pipeline. The red overlay marks the "
        f"{target_label} that will be removed. Analyze only the visible scene around "
        "the red target and return STRICT JSON, no markdown. The goal is to improve "
        "VOID video inpainting by marking nearby affected regions such as contact "
        "shadow, floor/wall texture disruption, reflection, edge halo, occlusion gaps, "
        "or small belongings attached to the removed target. Preserve neighboring "
        "people and objects. Use pixel values for a 256-768px image scale. Return this "
        "schema exactly: {"
        "\"scene_description\": string, "
        "\"prompt_hint\": string, "
        "\"affected_region_notes\": [string], "
        "\"contact_dilation_px\": integer, "
        "\"shadow_dilation_px\": integer, "
        "\"shadow_vertical_offset_px\": integer, "
        "\"shadow_horizontal_offset_px\": integer, "
        "\"confidence\": number"
        "}. Keep prompt_hint under 45 words."
    )


def _call_ollama_generate(
    *,
    base_url: str,
    model: str,
    prompt: str,
    images: list[str],
    timeout_seconds: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "format": "json",
        "keep_alive": os.environ.get("OLLAMA_VLM_KEEP_ALIVE", "0"),
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc}") from exc
    text = str(response_payload.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text


def _parse_ollama_analysis_response(response_text: str, model: str) -> VlmAnalysisResult:
    data = _extract_json_dict(response_text)
    notes = data.get("affected_region_notes") or []
    if not isinstance(notes, list):
        notes = [str(notes)]
    return VlmAnalysisResult(
        provider=VLM_PROVIDER_OLLAMA,
        model=model,
        scene_description=_clean_text(data.get("scene_description")),
        prompt_hint=_clean_text(data.get("prompt_hint")),
        affected_region_notes=tuple(_clean_text(note) for note in notes if _clean_text(note)),
        contact_dilation_px=_optional_int(data.get("contact_dilation_px")),
        shadow_dilation_px=_optional_int(data.get("shadow_dilation_px")),
        shadow_vertical_offset_px=_optional_int(data.get("shadow_vertical_offset_px")),
        shadow_horizontal_offset_px=_optional_int(data.get("shadow_horizontal_offset_px")),
        confidence=_bounded_float(data.get("confidence"), minimum=0.0, maximum=1.0),
        raw_response=response_text[:4000],
    )


def _extract_json_dict(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Ollama response did not contain a JSON object.")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama response JSON was not an object.")
    return parsed


def _resolve_ollama_base_url(base_url: str | None) -> str:
    value = (
        base_url
        or os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_OLLAMA_BASE_URL
    )
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")


def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        return default
    return max(minimum, min(parsed, maximum))


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _bounded_float(value, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(parsed, maximum))


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:500]
