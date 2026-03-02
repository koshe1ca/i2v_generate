from __future__ import annotations

from pathlib import Path
from typing import Optional

from models.settings import AppSettings
from services.pipeline_service import PipelineService
from services.pose_service import PoseService


class I2VController:
    def __init__(self, settings: AppSettings):
        self.s = settings
        self.pipe = PipelineService(settings)
        self.pose = PoseService(detector=settings.pose.detector)

    def extract_pose_from_video(self, ref_video: Path, out_pose_dir: Path) -> Path:
        res = self.pose.extract_pose_dir(
            ref_video_path=ref_video,
            pose_dir=out_pose_dir,
            size=self.s.pose.size,
            every_n=self.s.pose.every_n,
        )
        return res.pose_dir

    def generate_ad(self, prompt: str, pose_dir: Optional[Path] = None) -> Path:
        return self.pipe.generate_text2video_ad(
            prompt=prompt,
            out_name="ad_pose" if pose_dir else "ad",
            pose_dir=pose_dir,
        )