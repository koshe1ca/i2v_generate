# services/video_service.py
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np
import imageio
import time


class VideoService:
    def __init__(self, output_dir: str, mp4_crf: int = 18):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_crf = mp4_crf

    def save_frames(self, frames: List[Image.Image], tag: str) -> Path:
        out_dir = self.output_dir / "frames" / f"{tag}_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            fr.convert("RGB").save(out_dir / f"{i:05d}.png", compress_level=3)
        return out_dir

    def export_mp4(self, frames: List[Image.Image], fps: int, tag: str) -> Path:
        out_path = self.output_dir / f"{tag}_{int(time.time())}.mp4"
        np_frames = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]

        writer = imageio.get_writer(
            out_path,
            fps=int(fps),
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=[
                "-crf", str(int(self.mp4_crf)),
                "-preset", "slow",
                "-movflags", "+faststart",
            ],
        )
        try:
            for fr in np_frames:
                writer.append_data(fr)
        finally:
            writer.close()
        return out_path