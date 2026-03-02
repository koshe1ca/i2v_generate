# services/temporal_service.py
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2


class TemporalService:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _detect_face(self, img_bgr) -> Tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None
        # берем крупнейшее лицо
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return int(x), int(y), int(w), int(h)

    def apply_face_lock(self, frames: List[Image.Image], strength: float = 0.65) -> List[Image.Image]:
        if len(frames) < 2:
            return frames

        strength = float(np.clip(strength, 0.0, 1.0))

        out = [frames[0].convert("RGB")]
        prev = cv2.cvtColor(np.array(out[0]), cv2.COLOR_RGB2BGR)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        face_box = self._detect_face(prev)

        if face_box is None:
            # если лицо не нашли — просто легкая темпоральная сглаживалка по всему кадру
            return self.apply_global_temporal(frames, strength=strength)

        x, y, w, h = face_box
        roi_prev = prev[y:y+h, x:x+w].copy()

        for i in range(1, len(frames)):
            cur = cv2.cvtColor(np.array(frames[i].convert("RGB")), cv2.COLOR_RGB2BGR)
            cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # сдвиг ROI по среднему flow в области лица
            fx = np.mean(flow[y:y+h, x:x+w, 0])
            fy = np.mean(flow[y:y+h, x:x+w, 1])

            nx = int(np.clip(x + fx, 0, cur.shape[1] - w))
            ny = int(np.clip(y + fy, 0, cur.shape[0] - h))

            roi_cur = cur[ny:ny+h, nx:nx+w]
            if roi_cur.shape[:2] == roi_prev.shape[:2]:
                blended = cv2.addWeighted(roi_cur, 1.0 - strength, roi_prev, strength, 0)
                cur[ny:ny+h, nx:nx+w] = blended

            out_img = Image.fromarray(cv2.cvtColor(cur, cv2.COLOR_BGR2RGB))
            out.append(out_img)

            prev = cur
            prev_gray = cur_gray
            roi_prev = cur[ny:ny+h, nx:nx+w].copy()
            x, y = nx, ny

        return out

    def apply_global_temporal(self, frames: List[Image.Image], strength: float = 0.35) -> List[Image.Image]:
        if len(frames) < 2:
            return frames
        strength = float(np.clip(strength, 0.0, 1.0))
        out = [frames[0].convert("RGB")]
        prev = np.array(out[0], dtype=np.float32)

        for i in range(1, len(frames)):
            cur = np.array(frames[i].convert("RGB"), dtype=np.float32)
            blended = cur * (1.0 - strength) + prev * strength
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            out.append(Image.fromarray(blended))
            prev = blended.astype(np.float32)

        return out