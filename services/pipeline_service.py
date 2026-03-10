from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, ControlNetModel, DDIMScheduler, MotionAdapter

from models.history import HistoryDB, HistoryItem
from models.settings import AppSettings
from services.error_log_service import ErrorLogService
from services.face_restore_service import FaceRestoreService
from services.history_service import HistoryService
from services.pose_service import PoseService
from services.temporal_service import TemporalService
from services.video_service import VideoService

StageCallback = Callable[[str, int], None]
PreviewCallback = Callable[[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]], None]


class PipelineService:
    def __init__(self, settings: AppSettings):
        self.s = settings
        self.pipe: Optional[AnimateDiffPipeline] = None
        self.video = VideoService(str(self.s.effective_output_dir()), mp4_crf=self.s.video.mp4_crf, rife=self.s.rife)
        self.temporal = TemporalService()
        self.face_restore = FaceRestoreService(self.s.face_restore)
        self.pose_service = PoseService() if self.s.controlnet.enable else None
        self.history = HistoryService(str(self.s.effective_output_dir()))
        self.errors = ErrorLogService(str(self.s.effective_output_dir()))
        self._cancel = threading.Event()
        self._loaded_lora_paths: set[str] = set()
        self._ip_loaded = False
        self.s.refresh_duration()
        self.s.effective_output_dir().mkdir(parents=True, exist_ok=True)

    def cancel(self) -> None:
        self._cancel.set()

    def reset_cancel(self) -> None:
        self._cancel.clear()

    def load(self) -> None:
        if self.pipe is not None:
            return
        dtype = torch.float16 if self.s.torch_dtype in {"auto", "float16"} else torch.float32
        adapter = MotionAdapter.from_pretrained(self.s.motion_adapter, torch_dtype=dtype)
        controlnet = None
        if self.s.controlnet.enable:
            controlnet = ControlNetModel.from_pretrained(self.s.controlnet.model_id, torch_dtype=dtype)

        self.pipe = AnimateDiffPipeline.from_pretrained(
            self.s.base_model,
            motion_adapter=adapter,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.s.device)
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        self._load_loras()
        self._load_ip_adapter_if_needed()

    def _load_loras(self) -> None:
        if self.pipe is None:
            return
        active_names = []
        active_weights = []
        for item in self.s.lora.items:
            p = Path(item.path)
            if not item.enabled or not p.exists():
                continue
            adapter_name = p.stem
            if item.path not in self._loaded_lora_paths:
                self.pipe.load_lora_weights(item.path, adapter_name=adapter_name)
                self._loaded_lora_paths.add(item.path)
            active_names.append(adapter_name)
            active_weights.append(float(item.scale))
        if active_names:
            try:
                self.pipe.set_adapters(active_names, adapter_weights=active_weights)
            except Exception:
                # fallback for older diffusers
                try:
                    self.pipe.fuse_lora(lora_scale=float(active_weights[0]))
                except Exception:
                    pass

    def _load_ip_adapter_if_needed(self) -> None:
        if self.pipe is None or self._ip_loaded or not self.s.ip_adapter.enabled:
            return
        if not self.s.input_image and not self.s.ip_adapter.image_path:
            return
        if not hasattr(self.pipe, 'load_ip_adapter'):
            return
        try:
            if self.s.ip_adapter_enabled:
                self.pipe.load_ip_adapter(
                    self.s.ip_adapter.model_id,
                    subfolder=self.s.ip_adapter.subfolder,
                    weight_name=self.s.ip_adapter.weight_name,
                )
            if hasattr(self.pipe, 'set_ip_adapter_scale'):
                self.pipe.set_ip_adapter_scale(float(self.s.ip_adapter.scale))
            self._ip_loaded = True
        except Exception:
            # keep running without ip-adapter rather than fail hard
            self._ip_loaded = False

    def _load_pose_images(self, pose_dir: str) -> List[Image.Image]:
        p = Path(pose_dir)
        return [Image.open(x).convert('RGB') for x in sorted(p.glob('*.png'))]

    def _resolve_identity_image(self) -> Optional[Image.Image]:
        path = self.s.ip_adapter.image_path or self.s.input_image
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return Image.open(p).convert('RGB')

    def _make_previews(self, frames: List[Image.Image], item_id: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if not frames:
            return None, None, None
        preview_dir = self.s.effective_output_dir() / 'previews' / item_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        first = preview_dir / 'first.png'
        middle = preview_dir / 'middle.png'
        last = preview_dir / 'last.png'
        picks = [frames[0], frames[len(frames) // 2], frames[-1]]
        for src, dst in zip(picks, [first, middle, last]):
            src.convert('RGB').save(dst)
        return str(first), str(middle), str(last)

    def _callback_on_step_end(self, stage_cb: Optional[StageCallback], kwargs: dict):
        if self._cancel.is_set():
            raise RuntimeError('Generation cancelled by user')
        step_idx = kwargs.get('step_index', kwargs.get('i', 0))
        if stage_cb:
            stage_cb('Generating', int(step_idx) + 1)
        return kwargs

    def _build_kwargs(self, control_images=None, stage_cb=None):
        kwargs = {
            "prompt": self.s.prompt,
            "negative_prompt": self.s.negative_prompt,
            "num_frames": self.s.ad_num_frames,
            "num_inference_steps": self.s.ad_num_steps,
            "guidance_scale": self.s.ad_guidance,
            "width": self.s.ad_width,
            "height": self.s.ad_height,
            "generator": self._make_generator(),
        }

        if stage_cb is not None:
            kwargs["callback_on_step_end"] = stage_cb
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        if control_images is not None:
            kwargs["image"] = control_images

        if getattr(self, "_ip_adapter_image", None) is not None:
            kwargs["ip_adapter_image"] = self._ip_adapter_image

        return kwargs

    def _apply_post(self, frames: List[Image.Image], stage_cb: Optional[StageCallback], preview_cb: Optional[PreviewCallback]) -> List[Image.Image]:
        if self.s.temporal.enable:
            if stage_cb:
                stage_cb('Temporal stabilization', 0)
            if self.s.temporal.face_lock_only:
                frames = self.temporal.apply_face_lock(frames, strength=self.s.temporal.strength)
            else:
                frames = self.temporal.apply_global_temporal(frames, strength=self.s.temporal.strength)

        if self.s.face_restore.enable:
            if stage_cb:
                stage_cb('Face restore', 0)
            self.face_restore.settings = self.s.face_restore
            frames = self.face_restore.restore(
                frames,
                method=self.s.face_restore.backend if self.s.face_restore.backend != 'auto' else 'codeformer',
                strength=self.s.face_restore.strength,
            )

        if preview_cb and frames:
            preview_cb(frames[0], frames[len(frames) // 2], frames[-1])
        return frames

    def _prepare_motion_control(self, stage_cb: Optional[StageCallback]) -> tuple[Optional[List[Image.Image]], Optional[str]]:
        if self.s.mode != 'photo_plus_video_motion':
            return None, None
        if not self.s.ref_video:
            raise RuntimeError('Reference video is required for motion-transfer mode')
        if stage_cb:
            stage_cb('Extracting pose', 0)
        pose_dir = str((self.s.effective_output_dir() / 'pose_cache' / f'pose_{int(time.time())}').resolve())
        self.pose_service = self.pose_service or PoseService()
        self.pose_service.extract_pose_frames(self.s.ref_video, pose_dir)
        return self._load_pose_images(pose_dir), pose_dir

    def generate(self, stage_cb: Optional[StageCallback] = None, preview_cb: Optional[PreviewCallback] = None) -> Path:
        self.reset_cancel()
        try:
            if stage_cb:
                stage_cb('Loading models', 0)
            self.load()
            if self.pipe is None:
                raise RuntimeError('Pipeline not loaded')
            control_images, pose_dir = self._prepare_motion_control(stage_cb)
            if stage_cb:
                stage_cb('Generating', 0)
            out = self.pipe(**self._build_kwargs(control_images, stage_cb))
            frames = out.frames[0]
            if self._cancel.is_set():
                raise RuntimeError('Generation cancelled by user')

            frames = self._apply_post(frames, stage_cb, preview_cb)

            frames_dir = None
            if self.s.video.save_frames:
                if stage_cb:
                    stage_cb('Saving frames', 0)
                frames_dir = self.video.save_frames(frames, tag=self.s.mode)

            if stage_cb:
                stage_cb('Encoding MP4', 0)
            mp4 = self.video.export_mp4(frames, fps=self.s.video.fps, tag=self.s.mode)

            if self.s.rife.enable:
                if stage_cb:
                    stage_cb('RIFE interpolation', 0)
                mp4 = self.video.maybe_interpolate_with_rife(mp4, tag=self.s.mode)

            item_id = self.history.new_id()
            p1, p2, p3 = self._make_previews(frames, item_id)
            item = HistoryItem(
                id=item_id,
                created_at=HistoryDB.now_iso(),
                mode=self.s.mode,
                prompt=self.s.prompt,
                negative_prompt=self.s.negative_prompt,
                input_image=self.s.input_image,
                ref_video=self.s.ref_video,
                output_video=str(mp4),
                output_frames_dir=str(frames_dir) if frames_dir else None,
                loras=[x.model_dump() for x in self.s.lora.items],
                preview_first=p1,
                preview_middle=p2,
                preview_last=p3,
                settings=self.s.model_dump(),
            )
            self.history.append(item)
            if stage_cb:
                stage_cb('Done', self.s.video.steps)
            return Path(mp4)
        except Exception as exc:
            self.errors.log_exception(exc, context='PipelineService.generate')
            raise
