from services.pipeline_service import PipelineService, EngineConfig

def main():
    cfg = EngineConfig()
    service = PipelineService(cfg)

    prompt = "a realistic portrait photo, natural skin, soft daylight, 35mm, cinematic"
    neg = "lowres, worst quality, bad anatomy, deformed, blurry, watermark, text"

    face_emb = "assets/faceid_emb.npy"

    pipe = service._get_ad_pipe()
    id_embeds = service._load_faceid_embeds(face_emb)

    out = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_frames=12,
        guidance_scale=7.0,
        num_inference_steps=20,
        width=512,
        height=512,
        ip_adapter_image_embeds=[id_embeds],
    )

    frames = out.frames[0]
    mp4 = service._export_mp4(frames, fps=24, out_name="ad_faceid")
    print("saved:", mp4)

if __name__ == "__main__":
    main()