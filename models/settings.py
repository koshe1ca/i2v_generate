from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class AppSettings(BaseModel):
    # Paths
    output_dir: str = "outputs"
    assets_dir: str = "assets"

    # Runtime
    device: str = "auto"          # auto/cuda/cpu
    torch_dtype: str = "auto"     # auto/float16/float32

    # AnimateDiff
    ad_base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    ad_motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5"
    ad_num_frames: int = 16
    ad_num_steps: int = 25
    ad_guidance: float = 7.5
    ad_seed: int = 42
    ad_width: int = 512
    ad_height: int = 512

    # ControlNet/OpenPose (будем подключать)
    use_controlnet_pose: bool = False
    controlnet_openpose_id: str = "lllyasviel/control_v11p_sd15_openpose"
    pose_dir: Optional[str] = None

    # LoRA (позже подключим, но поле уже есть)
    lora_path: Optional[str] = None
    lora_scale: float = 0.8

    # Video export
    fps: int = 24
    mp4_crf: int = 18

    class Config:
        extra = "ignore"