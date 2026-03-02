from services.pipeline_service import PipelineService, EngineConfig

def main():
    cfg = EngineConfig(
        total_frames=360,   # 15 секунд при 24fps = 360 кадров
        chunk_frames=12,
        overlap=2,
        num_steps=20,       # на слабом ПК будет долго, на 5080 сильно быстрее
        guidance=7.0,
    )
    service = PipelineService(cfg)

    prompt = "a realistic portrait photo, natural skin, soft daylight, 35mm, cinematic"
    neg = "lowres, worst quality, bad anatomy, deformed, blurry, watermark, text"

    mp4 = service.generate_ad_faceid_long(
        prompt=prompt,
        negative_prompt=neg,
        faceid_emb_npy="assets/faceid_emb.npy",
        out_name="ad_faceid_15s",
        seed=42,
    )
    print("saved:", mp4)

if __name__ == "__main__":
    main()