from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class VideoSettings(BaseModel):
    num_frames: int = 24
    steps: int = 30
    guidance: float = 6.5
    width: int = 512
    height: int = 768
    fps: int = 12
    seed: int = 42
    mp4_crf: int = 18
    save_frames: bool = True


class TemporalSettings(BaseModel):
    enable: bool = True
    face_lock_only: bool = True
    strength: float = 0.75


class FaceRestoreSettings(BaseModel):
    enable: bool = True
    backend: Literal["auto", "codeformer", "gfpgan", "none"] = "auto"
    strength: float = 0.55


class ControlNetSettings(BaseModel):
    enable: bool = False
    model_id: str = "lllyasviel/control_v11p_sd15_openpose"
    conditioning_scale: float = 0.8


class RifeSettings(BaseModel):
    enable: bool = True
    target_fps: int = 24
    cli_path: Optional[str] = None


class IPAdapterSettings(BaseModel):
    enabled: bool = True
    model_id: str = "h94/IP-Adapter"
    subfolder: str = "models"
    weight_name: str = "ip-adapter_sd15.bin"
    image_path: Optional[str] = None
    scale: float = 0.6


class LoraItem(BaseModel):
    name: str = "LoRA"
    path: str
    scale: float = 0.75
    enabled: bool = True


class LoraSettings(BaseModel):
    items: List[LoraItem] = Field(default_factory=list)


class LongVideoSettings(BaseModel):
    enabled: bool = True
    target_duration_sec: float = 10.0
    chunk_frames: int = 16
    overlap_frames: int = 4
    stitch_blend: bool = True
    stitch_blend_strength: float = 1.0
    export_intermediate_chunks: bool = False


class AppSettings(BaseModel):
    mode: Literal["photo_prompt", "photo_plus_video_motion"] = "photo_prompt"
    quality_preset: Literal["fast", "balanced", "high"] = "balanced"

    base_model: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5"

    prompt: str = "masterpiece, realistic human, subtle head motion, smartphone portrait video"
    negative_prompt: str = "blurry, deformed, artifacts, flicker, ghosting, duplicate face"

    input_image: Optional[str] = None
    ref_video: Optional[str] = None
    output_override_dir: Optional[str] = None

    device: str = "cuda"
    torch_dtype: Literal["auto", "float16", "float32"] = "auto"

    video: VideoSettings = Field(default_factory=VideoSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    face_restore: FaceRestoreSettings = Field(default_factory=FaceRestoreSettings)
    controlnet: ControlNetSettings = Field(default_factory=ControlNetSettings)
    rife: RifeSettings = Field(default_factory=RifeSettings)
    ip_adapter: IPAdapterSettings = Field(default_factory=IPAdapterSettings)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    long_video: LongVideoSettings = Field(default_factory=LongVideoSettings)

    def refresh_duration(self) -> None:
        # keep target duration in sync with direct frame/fps edits when long mode is off
        if not self.long_video.enabled and self.video.fps > 0:
            self.long_video.target_duration_sec = max(1.0, self.video.num_frames / float(self.video.fps))

    def effective_output_dir(self) -> Path:
        return Path(self.output_override_dir or (Path.home() / ".i2v_generate_app" / "outputs"))

    def apply_preset(self, preset: str) -> None:
        preset = preset.lower().strip()
        self.quality_preset = preset  # type: ignore[assignment]
        if preset == "fast":
            self.video.num_frames = 16
            self.video.steps = 24
            self.video.guidance = 6.0
            self.video.width = 512
            self.video.height = 768
            self.video.fps = 12
            self.temporal.enable = True
            self.temporal.face_lock_only = True
            self.temporal.strength = 0.70
            self.face_restore.enable = True
            self.face_restore.strength = 0.45
            self.rife.enable = False
            self.rife.target_fps = 12
            self.ip_adapter.scale = 0.55
            self.long_video.chunk_frames = 12
            self.long_video.overlap_frames = 3
        elif preset == "high":
            self.video.num_frames = 20
            self.video.steps = 34
            self.video.guidance = 6.8
            self.video.width = 576
            self.video.height = 1024
            self.video.fps = 12
            self.temporal.enable = True
            self.temporal.face_lock_only = True
            self.temporal.strength = 0.80
            self.face_restore.enable = True
            self.face_restore.strength = 0.60
            self.rife.enable = True
            self.rife.target_fps = 24
            self.ip_adapter.scale = 0.65
            self.long_video.chunk_frames = 20
            self.long_video.overlap_frames = 5
        else:  # balanced
            self.video.num_frames = 16
            self.video.steps = 30
            self.video.guidance = 6.5
            self.video.width = 512
            self.video.height = 768
            self.video.fps = 12
            self.temporal.enable = True
            self.temporal.face_lock_only = True
            self.temporal.strength = 0.75
            self.face_restore.enable = True
            self.face_restore.strength = 0.55
            self.rife.enable = True
            self.rife.target_fps = 24
            self.ip_adapter.scale = 0.60
            self.long_video.chunk_frames = 16
            self.long_video.overlap_frames = 4
