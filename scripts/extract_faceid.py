from pathlib import Path
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

def main():
    img_path = Path("assets/input.jpg")  # поменяешь на своё
    out_path = Path("assets/faceid_emb.npy")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img = np.array(Image.open(img_path).convert("RGB"))
    faces = app.get(img)

    if not faces:
        raise SystemExit("Лицо не найдено. Возьми фото где лицо ближе/четче/анфас.")

    # Берем самое большое лицо (если вдруг несколько)
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding  # (512,)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb.astype(np.float32))
    print("OK saved:", out_path, "shape=", emb.shape)

if __name__ == "__main__":
    main()