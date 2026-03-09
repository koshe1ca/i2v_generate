from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


QualityPreset = Literal["fast", "balanced", "high"]
EngineMode = Literal["photo_prompt", "photo_plus_video_motion"]


class LoraItem(BaseModel):
    name: str = "LoRA"
    path: str
    scale: float = 0.75
    enabled: bool = True


class LoraStackSettings(BaseModel):
    items: List[LoraItem] = Field(default_factory=list)


class ControlNetSettings(BaseModel):
    enable: bool = False
    model_id: str = "lllyasviel/control_v11p_sd15_openpose"
    conditioning_scale: float = 1.0


class TemporalSettings(BaseModel):
    enable: bool = True
    face_lock_only: bool = True
    strength: float = 0.65


class FaceRestoreSettings(BaseModel):
    enable: bool = True
    backend: Literal["auto", "codeformer", "gfpgan", "basic"] = "auto"
    executable: Optional[str] = None
    model_dir: Optional[str] = None
    fidelity: float = 0.55
    strength: float = 0.65


class IPAdapterSettings(BaseModel):
    enabled: bool = True
    model_id: str = "h94/IP-Adapter"
    subfolder: str = "models"
    weight_name: str = "ip-adapter_sd15.bin"
    scale: float = 0.6
    image_path: Optional[str] = None


class RifeSettings(BaseModel):
    enable: bool = False
    executable: Optional[str] = None
    target_fps: int = 24
    factor: int = 2


class VideoSettings(BaseModel):
    num_frames: int = 16
    steps: int = 25
    guidance: float = 7.5
    fps: int = 16
    width: int = 512
    height: int = 512
    seed: int = 42
    save_frames: bool = True
    mp4_crf: int = 18
    duration_seconds: float = 2.0


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_dir: str = "outputs"
    output_override_dir: Optional[str] = None
    assets_dir: str = "assets"
    device: str = "cuda"
    torch_dtype: str = "float16"

    base_model: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5"
    prompt: str = "masterpiece, realistic human, cinematic lighting"
    negative_prompt: str = "blurry, deformed, extra fingers, artifacts"
    input_image: Optional[str] = None
    ref_video: Optional[str] = None
    mode: EngineMode = "photo_prompt"
    quality_preset: QualityPreset = "balanced"

    controlnet: ControlNetSettings = Field(default_factory=ControlNetSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    face_restore: FaceRestoreSettings = Field(default_factory=FaceRestoreSettings)
    ip_adapter: IPAdapterSettings = Field(default_factory=IPAdapterSettings)
    rife: RifeSettings = Field(default_factory=RifeSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    lora: LoraStackSettings = Field(default_factory=LoraStackSettings)

    def effective_output_dir(self) -> Path:
        return Path(self.output_override_dir or self.output_dir)

    def refresh_duration(self) -> None:
        if self.video.fps > 0:
            self.video.duration_seconds = round(self.video.num_frames / self.video.fps, 3)

    def apply_preset(self, preset: QualityPreset) -> None:
        self.quality_preset = preset
        if preset == "fast":
            self.video.num_frames = 12
            self.video.steps = 18
            self.video.guidance = 6.5
            self.video.width = 512
            self.video.height = 512
            self.video.fps = 16
            self.temporal.enable = True
            self.temporal.face_lock_only = True
            self.temporal.strength = 0.55
            self.face_restore.enable = False
            self.face_restore.fidelity = 0.6
            self.ip_adapter.enabled = True
            self.ip_adapter.scale = 0.55
            self.rife.enable = False
            self.rife.target_fps = 16
        elif preset == "balanced":
            self.video.num_frames = 16
            self.video.steps = 25
            self.video.guidance = 7.5
            self.video.width = 512
            self.video.height = 512
            self.video.fps = 24
            self.temporal.enable = True
            self.temporal.face_lock_only = True
            self.temporal.strength = 0.65
            self.face_restore.enable = True
            self.face_restore.fidelity = 0.55
            self.ip_adapter.enabled = True
            self.ip_adapter.scale = 0.6
            self.rife.enable = True
            self.rife.target_fps = 24
        elif preset == "high":
            self.video.num_frames = 24
            self.video.steps = 30
            self.video.guidance = 7.5
            self.video.width = 768
            self.video.height = 768
            self.video.fps = 24
            self.temporal.enable = True
            self.temporal.face_lock_only = False
            self.temporal.strength = 0.45
            self.face_restore.enable = True
            self.face_restore.fidelity = 0.5
            self.ip_adapter.enabled = True
            self.ip_adapter.scale = 0.68
            self.rife.enable = True
            self.rife.target_fps = 30
        self.refresh_duration()
