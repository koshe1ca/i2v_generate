from pathlib import Path
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


class FaceIdService:
    def __init__(self, providers=None):
        # пока CPU, чтобы не упираться в CUDA DLL
        self.providers = providers or ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def extract_embedding(self, image_path: str | Path) -> np.ndarray:
        img = np.array(Image.open(image_path).convert("RGB"))
        faces = self.app.get(img)
        if not faces:
            raise RuntimeError("Лицо не найдено")
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding.astype(np.float32)
        return emb

    def save_embedding(self, image_path: str | Path, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        emb = self.extract_embedding(image_path)
        np.save(out_path, emb)
        return out_path