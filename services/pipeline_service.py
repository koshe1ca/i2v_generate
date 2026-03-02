# services/pipeline_service.py
import json
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, ControlNetModel, DDIMScheduler

from models.settings import EngineSettings
from models.history import HistoryDB, HistoryItem
from services.video_service import VideoService
from services.temporal_service import TemporalService
from services.face_restore_service import FaceRestoreService


class PipelineService:
    def __init__(self, settings: EngineSettings):
        self.s = settings
        self.pipe: Optional[AnimateDiffPipeline] = None
        self.video = VideoService(self.s.output_dir, mp4_crf=self.s.video.mp4_crf)
        self.temporal = TemporalService()
        self.face_restore = FaceRestoreService()

        Path(self.s.output_dir).mkdir(parents=True, exist_ok=True)

    def load(self):
        adapter = MotionAdapter.from_pretrained(self.s.motion_adapter, torch_dtype=torch.float16)

        controlnet = None
        if self.s.controlnet.enable:
            controlnet = ControlNetModel.from_pretrained(self.s.controlnet.model_id, torch_dtype=torch.float16)

        self.pipe = AnimateDiffPipeline.from_pretrained(
            self.s.base_model,
            motion_adapter=adapter,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.s.device)

        # LoRA stack
        for item in self.s.lora.items:
            self.pipe.load_lora_weights(item.path)
            self.pipe.fuse_lora(lora_scale=float(item.scale))

    def _load_pose_images(self, pose_dir: str) -> List[Image.Image]:
        p = Path(pose_dir)
        imgs = [Image.open(x).convert("RGB") for x in sorted(p.glob("*.png"))]
        return imgs

    def generate_ad(self, pose_dir: Optional[str] = None) -> Path:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        control_images = None
        mode = "ad"
        if pose_dir:
            control_images = self._load_pose_images(pose_dir)
            mode = "ad_pose"

        out = self.pipe(
            prompt=self.s.prompt,
            negative_prompt=self.s.negative_prompt,
            num_frames=self.s.video.num_frames,
            num_inference_steps=self.s.video.steps,
            guidance_scale=self.s.video.guidance,
            width=self.s.video.width,
            height=self.s.video.height,
            control_image=control_images,
            controlnet_conditioning_scale=float(self.s.controlnet.conditioning_scale),
        )

        frames = out.frames[0]

        # --- Temporal consistency ---
        if self.s.temporal.enable:
            if self.s.temporal.face_lock_only:
                frames = self.temporal.apply_face_lock(frames, strength=self.s.temporal.strength)
            else:
                frames = self.temporal.apply_global_temporal(frames, strength=self.s.temporal.strength)

        # --- Face restore ---
        if self.s.face_restore.enable:
            frames = self.face_restore.restore(
                frames,
                method=self.s.face_restore.method,
                strength=self.s.face_restore.strength,
            )

        # save
        frames_dir = None
        if self.s.video.save_frames:
            frames_dir = self.video.save_frames(frames, tag=mode)

        mp4 = self.video.export_mp4(frames, fps=self.s.video.fps, tag=mode)

        # history
        self._append_history(
            HistoryItem(
                created_at=HistoryDB.now_iso(),
                mode=mode,
                prompt=self.s.prompt,
                negative_prompt=self.s.negative_prompt,
                input_image=self.s.input_image,
                ref_video=self.s.ref_video,
                pose_dir=pose_dir,
                output_video=str(mp4),
                output_frames_dir=str(frames_dir) if frames_dir else None,
                settings=self.s.model_dump(),
            )
        )

        return mp4

    def _append_history(self, item: HistoryItem):
        path = Path(self.s.output_dir) / "history.json"
        if path.exists():
            db = HistoryDB(**json.loads(path.read_text(encoding="utf-8")))
        else:
            db = HistoryDB()

        db.items.insert(0, item)
        path.write_text(db.model_dump_json(indent=2), encoding="utf-8")