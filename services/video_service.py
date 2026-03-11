from __future__ import annotations

from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
from PIL import Image


class VideoService:
    def __init__(self, output_dir: str, mp4_crf: int = 18, rife=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_crf = mp4_crf
        self.rife = rife

    def save_frames(self, frames: List[Image.Image], tag: str = "video") -> Path:
        frames_dir = self.output_dir / f"{tag}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(frames_dir / f"{i:05d}.png")
        return frames_dir

    def export_mp4(self, frames: List[Image.Image], fps: int = 12, tag: str = "video") -> Path:
        out_path = self.output_dir / f"{tag}_{self._stamp()}.mp4"
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
        try:
            for frame in frames:
                writer.append_data(np.array(frame.convert("RGB")))
        finally:
            writer.close()
        return out_path

    def maybe_interpolate_with_rife(self, mp4_path: Path, tag: str = "video") -> Path:
        return mp4_path

    def chunk_blend_sequence(self, frames: List[Image.Image], chunk_size: int, overlap: int, blend: bool = True) -> List[Image.Image]:
        if len(frames) <= chunk_size or overlap <= 0 or overlap >= chunk_size:
            return frames
        out: List[Image.Image] = []
        step = chunk_size - overlap
        chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), step)]
        if not chunks:
            return frames
        out.extend(chunks[0])
        for chunk in chunks[1:]:
            if len(chunk) <= overlap:
                break
            if blend and len(out) >= overlap:
                tail = out[-overlap:]
                head = chunk[:overlap]
                mixed = []
                for i in range(overlap):
                    a = i / max(1, overlap - 1)
                    img1 = np.array(tail[i]).astype(np.float32)
                    img2 = np.array(head[i]).astype(np.float32)
                    m = (img1 * (1.0 - a) + img2 * a).clip(0, 255).astype(np.uint8)
                    mixed.append(Image.fromarray(m))
                out[-overlap:] = mixed
            out.extend(chunk[overlap:])
        return out

    def _stamp(self) -> str:
        import time
        return str(int(time.time()))
