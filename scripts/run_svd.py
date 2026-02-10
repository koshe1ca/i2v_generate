from services.pipeline_service import PipelineService, EngineConfig

def main():
    cfg = EngineConfig(
        output_dir="outputs",
        device="auto",
        torch_dtype="auto",
        fps=24,
        svd_num_frames=8,       # для теста нормально
        svd_num_steps=20,        # быстрее
        svd_motion_bucket_id=127,
        svd_noise_aug_strength=0.02,
        svd_seed=42,
    )

    service = PipelineService(cfg)

    # ВАЖНО: укажи путь к своей фотке
    mp4_path = service.generate_image2video_svd(
        image="assets/input.jpg",
        resize_to=(1024, 576),   # SVD любит такое соотношение
        out_name="svd_test",
    )

    print("Saved:", mp4_path)

if __name__ == "__main__":
    main()