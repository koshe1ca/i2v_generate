import torch
from PIL import Image
import numpy as np

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

from insightface.app import FaceAnalysis

# --- config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
INPUT_FACE = "assets/input.jpg"   # положи сюда портрет
OUT_PATH = "outputs/faceid_sdxl.png"

PROMPT = "ultra realistic photo portrait, studio lighting, sharp focus, 85mm lens, natural skin texture"
NEG = "lowres, blurry, bad anatomy, deformed, extra fingers, watermark, text"

# --- 1) get face embedding via insightface ---
img = np.array(Image.open(INPUT_FACE).convert("RGB"))
fa = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))
faces = fa.get(img)
if not faces:
    raise RuntimeError("No face detected in assets/input.jpg")
face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])  # biggest face
emb = torch.tensor(face.normed_embedding, device=DEVICE).unsqueeze(0)

# --- 2) load SDXL pipeline ---
pipe = AutoPipelineForText2Image.from_pretrained(
    SDXL_BASE,
    torch_dtype=DTYPE,
    variant="fp16" if DTYPE == torch.float16 else None,
)
pipe = pipe.to(DEVICE)

# --- IMPORTANT ---
# IP-Adapter FaceID weights:
# we will use official HF repo from "h94/IP-Adapter"
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-faceid_sdxl.bin",
)
pipe.set_ip_adapter_scale(0.8)

# --- generate ---
image = pipe(
    prompt=PROMPT,
    negative_prompt=NEG,
    num_inference_steps=30,
    guidance_scale=6.0,
    ip_adapter_image_embeds=emb,
).images[0]

image.save(OUT_PATH)
print("Saved:", OUT_PATH)