from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from rich.console import Console

import imageio

# Diffusers
from diffusers import (
    MotionAdapter,
    AnimateDiffPipeline,
    DDIMScheduler,
    StableVideoDiffusionPipeline,
)

console = Console()


@dataclass(frozen=True)
class EngineConfig:
    # ---- Output ----
    output_dir: str = "outputs"

    # ---- Device ----
    device: str = "auto"
    torch_dtype: str = "auto"

    # ---- AnimateDiff (Text2Video) ----
    ad_base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    ad_motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5"
    ad_num_frames: int = 8
    ad_num_steps: int = 10
    ad_guidance: float = 7.5
    ad_seed: int = 42

    # ---- SVD (Image2Video) ----
    svd_model_id: str = "stabilityai/stable-video-diffusion-img2vid"
    svd_num_frames: int = 14
    svd_num_steps: int = 35

    # меньше движения = меньше "плывёт" лицо
    svd_motion_bucket_id: int = 50

    # 0.0–0.01 лучше держит лицо
    svd_noise_aug_strength: float = 0.0

    # guidance в SVD есть, и он влияет на детали/стабильность
    svd_min_guidance: float = 1.5
    svd_max_guidance: float = 3.5

    svd_seed: int = 42

    # ---- Video ----
    fps: int = 24

    svd_num_frames: int = 48  # 2 секунды при 24 fps
    svd_num_steps: int = 30  # качество ↑, время ↑
    svd_motion_bucket_id: int = 50  # умеренное движение (лицо меньше плывёт)
    svd_noise_aug_strength: float = 0.0

    svd_min_guidance: float = 1.5
    svd_max_guidance: float = 3.5
    mp4_crf: int = 16
    save_frames: bool = True
    mp4_crf: int = 16  # лучше качество (файл больше)

    # ---- Frames export ----
    save_frames: bool = True


class PipelineService:
    """
    Service layer: loads pipelines once, runs generation, exports mp4.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self._device = self._resolve_device(cfg.device)
        self._dtype = self._resolve_dtype(cfg.torch_dtype, self._device)

        self._ad_pipe: Optional[AnimateDiffPipeline] = None
        self._svd_pipe: Optional[StableVideoDiffusionPipeline] = None

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        console.print(f"[green]Device:[/green] {self._device}")
        console.print(f"[green]Dtype:[/green] {self._dtype}")

    # ----------------------------
    # Public API
    # ----------------------------

    def generate_text2video_ad(
            self,
            prompt: str,
            negative_prompt: str = "bad quality, worst quality",
            num_frames: Optional[int] = None,
            width: int = 512,
            height: int = 512,
            fps: Optional[int] = None,
            seed: Optional[int] = None,
            out_name: Optional[str] = None,
    ) -> Path:
        """
        AnimateDiff (SD1.5) Text -> Video (mp4)
        """
        pipe = self._get_ad_pipe()

        n_frames = num_frames if num_frames is not None else self.cfg.ad_num_frames
        fps = fps if fps is not None else self.cfg.fps
        seed = seed if seed is not None else self.cfg.ad_seed

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        console.print("[cyan]Running AnimateDiff Text→Video...[/cyan]")
        t0 = time.time()

        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=int(n_frames),
            guidance_scale=float(self.cfg.ad_guidance),
            num_inference_steps=int(self.cfg.ad_num_steps),
            generator=generator,
            width=int(width),
            height=int(height),
        )

        frames = out.frames[0]
        mp4_path = self._export_mp4(frames, fps=fps, out_name=out_name or "ad_text2video")

        console.print(f"[green]Done[/green] in {time.time() - t0:.1f}s → {mp4_path}")
        return mp4_path

    def generate_image2video_svd(
            self,
            image: Union[str, Path, Image.Image],
            fps: Optional[int] = None,
            seed: Optional[int] = None,
            out_name: Optional[str] = None,
            resize_to: Tuple[int, int] = (1024, 576),
    ) -> Path:
        """
        Stable Video Diffusion (SVD) Image -> Video (mp4)
        """
        pipe = self._get_svd_pipe()

        fps = fps if fps is not None else self.cfg.fps
        seed = seed if seed is not None else self.cfg.svd_seed

        # 👉 вот здесь создаётся init_img
        init_img = self._load_image(image)
        init_img = init_img.convert("RGB")
        init_img = self._fit_center_crop(init_img, resize_to)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        console.print("[cyan]Running SVD Image→Video...[/cyan]")
        t0 = time.time()

        out = pipe(
            image=init_img,
            num_frames=int(self.cfg.svd_num_frames),
            num_inference_steps=int(self.cfg.svd_num_steps),
            generator=generator,
            motion_bucket_id=int(self.cfg.svd_motion_bucket_id),
            noise_aug_strength=float(self.cfg.svd_noise_aug_strength),
            min_guidance_scale=float(self.cfg.svd_min_guidance),
            max_guidance_scale=float(self.cfg.svd_max_guidance),
        )

        frames = self._normalize_frames(out.frames[0])

        if getattr(self.cfg, "save_frames", False):
            self._export_frames_png(frames, out_name=out_name or "svd_img2video")

        mp4_path = self._export_mp4(frames, fps=fps, out_name=out_name or "svd_img2video")

        console.print(f"[green]Done[/green] in {time.time() - t0:.1f}s → {mp4_path}")
        return mp4_path

    # ----------------------------
    # Pipeline loaders (lazy)
    # ----------------------------

    def _get_ad_pipe(self) -> AnimateDiffPipeline:
        if self._ad_pipe is not None:
            return self._ad_pipe

        console.print("[yellow]Loading AnimateDiff pipelines...[/yellow]")
        adapter = MotionAdapter.from_pretrained(self.cfg.ad_motion_adapter_id, torch_dtype=self._dtype)

        pipe = AnimateDiffPipeline.from_pretrained(
            self.cfg.ad_base_model_id,
            motion_adapter=adapter,
            torch_dtype=self._dtype,
            safety_checker=None,  # local, faster (you can remove if you want)
        )

        # Scheduler: important recommendation for AnimateDiff
        pipe.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ad_base_model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )

        self._configure_pipe(pipe)
        self._ad_pipe = pipe
        return pipe

    def _export_frames_png(self, frames: List[Image.Image], out_name: str) -> Path:
        out_dir = Path(self.cfg.output_dir) / "frames" / f"{out_name}_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, fr in enumerate(frames):
            fr = fr.convert("RGB")
            fr.save(out_dir / f"{i:04d}.png", compress_level=3)

        console.print(f"[green]Saved frames:[/green] {out_dir}")
        return out_dir

    def _get_svd_pipe(self) -> StableVideoDiffusionPipeline:
        if self._svd_pipe is not None:
            return self._svd_pipe

        console.print("[yellow]Loading SVD pipeline...[/yellow]")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.cfg.svd_model_id,
            torch_dtype=self._dtype,
            safety_checker=None,
        )

        self._configure_pipe(pipe)
        self._svd_pipe = pipe
        return pipe

    # ----------------------------
    # Helpers
    # ----------------------------

    def _configure_pipe(self, pipe):
        # Memory savers (important on MPS too)
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        # Move to device
        if self._device in ("cuda", "mps"):
            pipe.to(self._device)
        else:
            pipe.to("cpu")

    def _export_mp4(self, frames: List[Image.Image], fps: int, out_name: str) -> Path:
        out_dir = Path(self.cfg.output_dir)
        out_path = out_dir / f"{out_name}_{int(time.time())}.mp4"

        # Convert PIL -> uint8 np arrays
        np_frames = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]

        # Write MP4 with ffmpeg backend
        writer = imageio.get_writer(
            out_path,
            fps=int(fps),
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            ffmpeg_params=[
                "-crf", str(int(self.cfg.mp4_crf)),
                "-preset", "slow",
                "-movflags", "+faststart",
            ]
        )
        try:
            for fr in np_frames:
                writer.append_data(fr)
        finally:
            writer.close()

        return out_path

    def _fit_center_crop(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        target_w, target_h = size
        img = img.convert("RGB")

        src_w, src_h = img.size
        src_ratio = src_w / src_h
        tgt_ratio = target_w / target_h

        # сначала масштабируем так, чтобы перекрыть нужную область
        if src_ratio > tgt_ratio:
            # исходник шире — подгоняем по высоте
            new_h = target_h
            new_w = int(new_h * src_ratio)
        else:
            # исходник выше — подгоняем по ширине
            new_w = target_w
            new_h = int(new_w / src_ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))


    def _normalize_frames(self, frames) -> List[Image.Image]:
        # frames might be list[np.ndarray] or list[PIL.Image]
        normalized: List[Image.Image] = []
        for f in frames:
            if isinstance(f, Image.Image):
                normalized.append(f)
            else:
                normalized.append(Image.fromarray(np.asarray(f).astype(np.uint8)))
        return normalized

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p)

    def _resolve_device(self, device: str) -> str:
        device = (device or "auto").lower()
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        # MPS available?
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, dtype: str, device: str):
        dtype = (dtype or "auto").lower()
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if dtype == "float32":
            return torch.float32

        # auto
        if device in ("cuda", "mps"):
            return torch.float16
        return torch.float32