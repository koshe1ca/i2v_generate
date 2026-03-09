from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

from models.settings import FaceRestoreSettings


class FaceRestoreService:
    """Best-effort face restoration.

    Priority:
    1) external CodeFormer CLI if available
    2) external GFPGAN CLI if available
    3) lightweight basic fallback enhancement
    """

    def __init__(self, settings: Optional[FaceRestoreSettings] = None):
        self.settings = settings or FaceRestoreSettings()
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def restore(self, frames: List[Image.Image], method: str = 'codeformer', strength: float = 0.65) -> List[Image.Image]:
        if not frames:
            return frames
        backend = self.settings.backend
        if backend == 'auto':
            backend = method or 'codeformer'
        fidelity = float(np.clip(getattr(self.settings, 'fidelity', 0.55), 0.0, 1.0))

        if backend == 'codeformer':
            restored = self._run_codeformer(frames, fidelity=fidelity)
            if restored is not None:
                return self._blend_frames(frames, restored, strength)
        if backend == 'gfpgan':
            restored = self._run_gfpgan(frames)
            if restored is not None:
                return self._blend_frames(frames, restored, strength)
        return [self._basic_restore(fr, strength=strength) for fr in frames]

    def _blend_frames(self, original: List[Image.Image], restored: List[Image.Image], strength: float) -> List[Image.Image]:
        out: List[Image.Image] = []
        alpha = float(np.clip(strength, 0.0, 1.0))
        for src, dst in zip(original, restored):
            out.append(Image.blend(src.convert('RGB'), dst.convert('RGB'), alpha))
        return out

    def _detect_faces(self, arr_bgr: np.ndarray):
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
        return self._face_cascade.detectMultiScale(gray, 1.1, 5)

    def _basic_restore(self, frame: Image.Image, strength: float = 0.65) -> Image.Image:
        rgb = frame.convert('RGB')
        arr = np.array(rgb)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        faces = self._detect_faces(bgr)
        if len(faces) == 0:
            return rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

        out = bgr.copy()
        for (x, y, w, h) in faces:
            roi = out[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            den = cv2.fastNlMeansDenoisingColored(roi, None, 3, 3, 7, 21)
            sharp = cv2.GaussianBlur(den, (0, 0), 1.0)
            sharp = cv2.addWeighted(den, 1.35 + strength * 0.3, sharp, -(0.35 + strength * 0.15), 0)
            out[y:y+h, x:x+w] = sharp
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)

    def _run_codeformer(self, frames: List[Image.Image], fidelity: float = 0.55) -> Optional[List[Image.Image]]:
        exe = self._resolve_executable(['inference_codeformer.py', 'codeformer'])
        if exe is None:
            return None
        with tempfile.TemporaryDirectory(prefix='i2v_cf_in_') as inp, tempfile.TemporaryDirectory(prefix='i2v_cf_out_') as out:
            inp_p = Path(inp); out_p = Path(out)
            for i, fr in enumerate(frames):
                fr.convert('RGB').save(inp_p / f'{i:05d}.png')
            cmd = []
            if exe.endswith('.py'):
                python_bin = shutil.which('python') or shutil.which('python3')
                if python_bin is None:
                    return None
                cmd = [python_bin, exe, '-i', str(inp_p), '-o', str(out_p), '--w', str(float(np.clip(fidelity, 0.0, 1.0)))]
            else:
                cmd = [exe, '-i', str(inp_p), '-o', str(out_p), '--w', str(float(np.clip(fidelity, 0.0, 1.0)))]
            model_dir = self.settings.model_dir
            if model_dir:
                cmd += ['--model_dir', model_dir]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                return None
            restored = sorted((out_p).rglob('*.png'))
            if not restored:
                restored = sorted((out_p / 'final_results').glob('*.png'))
            if not restored:
                return None
            return [Image.open(p).convert('RGB') for p in restored[:len(frames)]]

    def _run_gfpgan(self, frames: List[Image.Image]) -> Optional[List[Image.Image]]:
        exe = self._resolve_executable(['inference_gfpgan.py', 'gfpgan'])
        if exe is None:
            return None
        with tempfile.TemporaryDirectory(prefix='i2v_gf_in_') as inp, tempfile.TemporaryDirectory(prefix='i2v_gf_out_') as out:
            inp_p = Path(inp); out_p = Path(out)
            for i, fr in enumerate(frames):
                fr.convert('RGB').save(inp_p / f'{i:05d}.png')
            if exe.endswith('.py'):
                python_bin = shutil.which('python') or shutil.which('python3')
                if python_bin is None:
                    return None
                cmd = [python_bin, exe, '-i', str(inp_p), '-o', str(out_p)]
            else:
                cmd = [exe, '-i', str(inp_p), '-o', str(out_p)]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                return None
            restored = sorted(out_p.rglob('*.png'))
            if not restored:
                return None
            return [Image.open(p).convert('RGB') for p in restored[:len(frames)]]

    def _resolve_executable(self, names: List[str]) -> Optional[str]:
        explicit = self.settings.executable
        if explicit and Path(explicit).exists():
            return explicit
        for key in ('CODEFORMER_EXE', 'GFPGAN_EXE'):
            env = os.environ.get(key)
            if env and Path(env).exists():
                return env
        for name in names:
            if Path(name).exists():
                return str(Path(name).resolve())
            found = shutil.which(name)
            if found:
                return found
        return None
