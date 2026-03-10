from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class VideoService:
    def __init__(self, output_dir: str, mp4_crf: int = 18, rife=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_crf = mp4_crf
        self.rife = rife

    def _tag_dir(self, tag: str) -> Path:
        p = self.output_dir / tag
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_frames(self, frames: List[Image.Image], tag: str) -> str:
        ts = int(time.time())
        out_dir = self._tag_dir(f"frames_{tag}_{ts}")
        for idx, frame in enumerate(frames):
            frame.convert("RGB").save(out_dir / f"{idx:05d}.png")
        return str(out_dir)

    def export_mp4(self, frames: List[Image.Image], fps: int, tag: str) -> str:
        ts = int(time.time())
        out_path = self.output_dir / f"{tag}_{ts}.mp4"
        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", str(self.mp4_crf)],
        )
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame.convert("RGB")))
        finally:
            writer.close()
        return str(out_path)

    def blend_frame_lists(self, prev_frames: List[Image.Image], next_frames: List[Image.Image], overlap: int) -> List[Image.Image]:
        if not prev_frames:
            return list(next_frames)
        if overlap <= 0:
            return prev_frames + next_frames
        overlap = min(overlap, len(prev_frames), len(next_frames))
        head = prev_frames[:-overlap]
        tail_prev = prev_frames[-overlap:]
        tail_next = next_frames[:overlap]
        blended: List[Image.Image] = []
        for idx in range(overlap):
            alpha = (idx + 1) / float(overlap + 1)
            a = np.asarray(tail_prev[idx].convert("RGB")).astype(np.float32)
            b = np.asarray(tail_next[idx].convert("RGB")).astype(np.float32)
            c = np.clip(a * (1.0 - alpha) + b * alpha, 0, 255).astype(np.uint8)
            blended.append(Image.fromarray(c))
        return head + blended + next_frames[overlap:]

    def save_chunk_preview(self, frames: List[Image.Image], tag: str, chunk_index: int) -> Optional[str]:
        if not frames:
            return None
        folder = self._tag_dir(f"chunk_previews_{tag}")
        out = folder / f"chunk_{chunk_index:03d}.png"
        frames[len(frames) // 2].convert("RGB").save(out)
        return str(out)

    def maybe_interpolate_with_rife(self, input_mp4: str, tag: str) -> str:
        if not self.rife or not getattr(self.rife, "enable", False):
            return input_mp4
        cli_path = getattr(self.rife, "cli_path", None)
        target_fps = int(getattr(self.rife, "target_fps", 24))
        if not cli_path:
            return input_mp4
        cli = Path(cli_path)
        if not cli.exists():
            return input_mp4
        output_mp4 = str(self.output_dir / f"{tag}_rife_{int(time.time())}.mp4")
        try:
            subprocess.run(
                [str(cli), "-i", input_mp4, "-o", output_mp4, "-f", str(target_fps)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return output_mp4
        except Exception:
            return input_mp4
