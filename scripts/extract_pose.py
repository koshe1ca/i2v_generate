# scripts/extract_pose.py
import argparse
from services.pose_service import PoseService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-video", required=True)
    parser.add_argument("--out", default="outputs/pose")

    args = parser.parse_args()

    service = PoseService()
    service.extract_pose_frames(args.ref_video, args.out)


if __name__ == "__main__":
    main()