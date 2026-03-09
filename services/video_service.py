from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
from PIL import Image

from models.settings import RifeSettings


class VideoService:
    def __init__(self, output_dir: str, mp4_crf: int = 18, rife: Optional[RifeSettings] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mp4_crf = mp4_crf
        self.rife = rife or RifeSettings()

    def save_frames(self, frames: List[Image.Image], tag: str) -> Path:
        out_dir = self.output_dir / 'frames' / f'{tag}_{int(time.time())}'
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            fr.convert('RGB').save(out_dir / f'{i:05d}.png', compress_level=3)
        return out_dir

    def export_mp4(self, frames: List[Image.Image], fps: int, tag: str) -> Path:
        out_path = self.output_dir / f'{tag}_{int(time.time())}.mp4'
        np_frames = [np.array(f.convert('RGB'), dtype=np.uint8) for f in frames]
        writer = imageio.get_writer(
            out_path,
            fps=int(fps),
            codec='libx264',
            pixelformat='yuv420p',
            ffmpeg_params=['-crf', str(int(self.mp4_crf)), '-preset', 'slow', '-movflags', '+faststart'],
        )
        try:
            for fr in np_frames:
                writer.append_data(fr)
        finally:
            writer.close()
        return out_path

    def maybe_interpolate_with_rife(self, input_mp4: str | Path, tag: str) -> Path:
        if not self.rife.enable:
            return Path(input_mp4)
        exe = self._resolve_rife_executable()
        if exe is None:
            return Path(input_mp4)
        output_mp4 = self.output_dir / f'{tag}_rife_{int(time.time())}.mp4'
        self.interpolate_with_rife(input_mp4, output_mp4, self.rife.target_fps, exe)
        return output_mp4 if output_mp4.exists() else Path(input_mp4)

    def interpolate_with_rife(self, input_mp4: str | Path, output_mp4: str | Path, target_fps: int, executable: Optional[str] = None) -> Path:
        exe = executable or self._resolve_rife_executable()
        if exe is None:
            raise RuntimeError('RIFE executable not found. Set settings.rife.executable or RIFE_EXE env var.')
        input_mp4 = Path(input_mp4)
        output_mp4 = Path(output_mp4)
        output_mp4.parent.mkdir(parents=True, exist_ok=True)

        # Supports common CLIs: rife-ncnn-vulkan and python wrappers.
        # We export frames first for the widest compatibility.
        with tempfile.TemporaryDirectory(prefix='i2v_rife_') as tmp:
            tmp_p = Path(tmp)
            frames_in = tmp_p / 'in'
            frames_out = tmp_p / 'out'
            frames_in.mkdir(parents=True, exist_ok=True)
            frames_out.mkdir(parents=True, exist_ok=True)
            reader = imageio.get_reader(str(input_mp4))
            for i, fr in enumerate(reader):
                Image.fromarray(fr).save(frames_in / f'{i:05d}.png')
            reader.close()
            cmd = [exe]
            if 'rife-ncnn-vulkan' in Path(exe).name:
                cmd += ['-i', str(frames_in), '-o', str(frames_out), '-f', 'png', '-n', str(max(2, int(self.rife.factor)))]
            else:
                python_bin = shutil.which('python') or shutil.which('python3')
                if exe.endswith('.py') and python_bin:
                    cmd = [python_bin, exe, '--input', str(frames_in), '--output', str(frames_out), '--fps', str(int(target_fps))]
                else:
                    cmd += ['--input', str(frames_in), '--output', str(frames_out), '--fps', str(int(target_fps))]
            subprocess.run(cmd, check=True)
            out_frames = sorted(frames_out.glob('*.png'))
            if not out_frames:
                raise RuntimeError('RIFE finished but produced no output frames')
            imgs = [Image.open(p).convert('RGB') for p in out_frames]
            return self.export_mp4(imgs, fps=target_fps, tag=output_mp4.stem)

    def _resolve_rife_executable(self) -> Optional[str]:
        explicit = self.rife.executable
        if explicit and Path(explicit).exists():
            return explicit
        env = os.environ.get('RIFE_EXE') if 'os' in globals() else None
        if env and Path(env).exists():
            return env
        for name in ('rife-ncnn-vulkan', 'rife'):
            found = shutil.which(name)
            if found:
                return found
        return None
