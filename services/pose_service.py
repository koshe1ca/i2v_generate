# services/pose_service.py
from pathlib import Path
import cv2
from controlnet_aux import OpenposeDetector
from PIL import Image


class PoseService:
    def __init__(self):
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    def extract_pose_frames(self, video_path: str, out_dir: str):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = self.detector(img)

            pose.save(out_dir / f"{idx:05d}.png")
            idx += 1

        cap.release()
        return out_dir