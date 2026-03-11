from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from models.settings import AppSettings


class PhotoPreserveService:
    """
    First iteration of Path B:
    - keeps the original photo as the base for every frame
    - animates by geometric warping instead of regenerating full frames
    - preserves background by compositing a softly warped subject back onto the original image
    - can optionally derive motion from a reference video using face landmarks / optical flow-lite cues

    This is intentionally conservative: better to move less and preserve the photo than to generate a new scene.
    """

    def __init__(self, settings: AppSettings):
        self.s = settings
        self.pp = settings.photo_preserve
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def generate(self, stage_cb=None) -> List[Image.Image]:
        base = self._load_image(self.s.input_image)
        motion_curve = self._build_motion_curve(base)
        frames = self._animate(base, motion_curve)
        return frames

    def _load_image(self, path: Optional[str]) -> Image.Image:
        if not path:
            raise RuntimeError("Photo-preserve mode requires input_image")
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"Input image not found: {p}")
        img = Image.open(p).convert("RGB")
        if (img.width, img.height) != (self.s.video.width, self.s.video.height):
            img = img.resize((self.s.video.width, self.s.video.height), Image.LANCZOS)
        return img

    def _build_motion_curve(self, base: Image.Image) -> List[Tuple[float, float, float, float, float]]:
        total = self._target_total_frames()
        if self.s.mode == "photo_plus_video_motion" and self.s.ref_video:
            curve = self._curve_from_video(base, self.s.ref_video, total)
            if curve:
                return curve
        return self._curve_from_prompt(total)

    def _curve_from_prompt(self, total: int) -> List[Tuple[float, float, float, float, float]]:
        pp = self.pp
        curve: List[Tuple[float, float, float, float, float]] = []
        for i in range(total):
            t = i / max(1, total - 1)
            sway = math.sin(t * math.pi * 2.0) * pp.sway_amount
            breathe = math.sin(t * math.pi * 2.0) * pp.breathing_amount
            blink = 0.0
            if total > 8:
                # two soft blinks over the clip
                for center in (0.28, 0.72):
                    d = abs(t - center)
                    if d < 0.045:
                        blink = max(blink, (1 - d / 0.045) * pp.blink_amount)
            mouth = 0.0
            if "talk" in self.s.prompt.lower() or "speaking" in self.s.prompt.lower():
                mouth = abs(math.sin(t * math.pi * 6.0)) * pp.mouth_open_amount
            curve.append((sway, breathe, blink, mouth, 0.0))
        return curve

    def _curve_from_video(self, base: Image.Image, ref_video: str, total: int) -> Optional[List[Tuple[float, float, float, float, float]]]:
        cap = cv2.VideoCapture(str(ref_video))
        if not cap.isOpened():
            return None
        vals: List[Tuple[float, float, float, float, float]] = []
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        base_box = self._detect_face_bbox(np.array(base)[:, :, ::-1])
        last_box = None
        while ok and len(vals) < total:
            box = self._detect_face_bbox(frame)
            if box is None and last_box is not None:
                box = last_box
            if box is not None and base_box is not None:
                dx = (box[0] - base_box[0]) / max(1.0, base_box[2])
                dy = (box[1] - base_box[1]) / max(1.0, base_box[3])
                scale = (box[2] / max(1.0, base_box[2])) - 1.0
                vals.append((dx * 0.2, dy * 0.2, 0.0, 0.0, scale * 0.2))
                last_box = box
            ok, frame = cap.read()
        cap.release()
        if not vals:
            return None
        # resample to requested length
        out: List[Tuple[float, float, float, float, float]] = []
        for i in range(total):
            idx = int(round((i / max(1, total - 1)) * (len(vals) - 1)))
            out.append(vals[idx])
        return out

    def _detect_face_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h)

    def _target_total_frames(self) -> int:
        if self.s.long_video.enabled:
            return max(self.s.video.num_frames, int(round(self.s.long_video.target_duration_sec * self.s.video.fps)))
        return self.s.video.num_frames

    def _animate(self, base: Image.Image, curve: Sequence[Tuple[float, float, float, float, float]]) -> List[Image.Image]:
        base_np = np.array(base)
        bgr = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR)
        mask = self._subject_mask(bgr)
        face_box = self._detect_face_bbox(bgr)
        frames: List[Image.Image] = []
        for sway, breathe, blink, mouth, scale in curve:
            warped = self._warp_subject(base_np, mask, face_box, sway, breathe, blink, mouth, scale)
            frames.append(Image.fromarray(warped))
        return frames

    def _subject_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        # conservative foreground mask around the person. First version uses center-biased ellipse + face expansion.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.22), int(h * 0.38)), 0, 0, 360, 255, -1)
        face_box = self._detect_face_bbox(bgr)
        if face_box is not None:
            x, y, fw, fh = face_box
            grow = int(max(fw, fh) * 1.25)
            cx, cy = x + fw // 2, y + fh // 2
            cv2.ellipse(mask, (cx, cy), (grow, int(grow * 1.2)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=11, sigmaY=11)
        return mask

    def _warp_subject(
        self,
        base_rgb: np.ndarray,
        mask: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]],
        sway: float,
        breathe: float,
        blink: float,
        mouth: float,
        scale: float,
    ) -> np.ndarray:
        h, w = base_rgb.shape[:2]
        dx = int(round(sway * w * self.pp.warp_strength))
        dy = int(round(breathe * h * self.pp.warp_strength))
        subject = base_rgb.copy()
        M = np.float32([[1.0 + scale * 0.05, 0, dx], [0, 1.0 + scale * 0.03, dy]])
        warped = cv2.warpAffine(subject, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if face_box is not None:
            x, y, fw, fh = face_box
            warped = self._face_micro_motion(warped, x, y, fw, fh, blink, mouth, sway)

        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        if not self.pp.keep_background:
            return warped
        out = (warped.astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return out

    def _face_micro_motion(self, rgb: np.ndarray, x: int, y: int, fw: int, fh: int, blink: float, mouth: float, sway: float) -> np.ndarray:
        out = rgb.copy()
        face = out[y:y + fh, x:x + fw].copy()
        if face.size == 0:
            return out
        # blink by vertically compressing eye strip
        eye_y1 = int(fh * 0.28)
        eye_y2 = int(fh * 0.46)
        if blink > 0:
            eye = face[eye_y1:eye_y2]
            target_h = max(1, int(eye.shape[0] * (1.0 - blink * 0.85)))
            eye2 = cv2.resize(eye, (eye.shape[1], target_h), interpolation=cv2.INTER_LINEAR)
            pad_top = (eye.shape[0] - target_h) // 2
            pad_bot = eye.shape[0] - target_h - pad_top
            eye2 = cv2.copyMakeBorder(eye2, pad_top, pad_bot, 0, 0, borderType=cv2.BORDER_REPLICATE)
            face[eye_y1:eye_y2] = eye2[: eye.shape[0], : eye.shape[1]]
        # mouth by scaling the lower middle strip
        mouth_y1 = int(fh * 0.63)
        mouth_y2 = int(fh * 0.82)
        if mouth > 0:
            area = face[mouth_y1:mouth_y2].copy()
            target_h = max(1, int(area.shape[0] * (1.0 + mouth * 0.6)))
            area2 = cv2.resize(area, (area.shape[1], target_h), interpolation=cv2.INTER_LINEAR)
            if target_h >= area.shape[0]:
                start = (target_h - area.shape[0]) // 2
                area2 = area2[start:start + area.shape[0]]
            else:
                pad = area.shape[0] - target_h
                area2 = cv2.copyMakeBorder(area2, pad // 2, pad - pad // 2, 0, 0, borderType=cv2.BORDER_REPLICATE)
            face[mouth_y1:mouth_y2] = area2[: area.shape[0], : area.shape[1]]
        # tiny horizontal face drift
        drift = int(round(sway * fw * 0.08))
        face = np.roll(face, drift, axis=1)
        out[y:y + fh, x:x + fw] = face
        return out
