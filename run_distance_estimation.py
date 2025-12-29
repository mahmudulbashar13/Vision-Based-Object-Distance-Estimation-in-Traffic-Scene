#!/usr/bin/env python3
"""CLI runner for vision-based object distance estimation in traffic scenes."""
import argparse
from src.distance_estimation.pipeline import DistanceEstimationPipeline


def main():
    p = argparse.ArgumentParser(description="Detect, track, and estimate distances in traffic videos")
    p.add_argument("--yolo-weights", required=True, help="Path to YOLO11/YOLOv8 weights file")
    p.add_argument("--depth-type", choices=["midas", "custom"], default="midas", help="Depth model type")
    p.add_argument("--depth-checkpoint", default=None, help="Path to Depth Anything v2 checkpoint (required for custom)")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--out-video", default="output_annotated.mp4", help="Annotated output video")
    p.add_argument("--out-csv", default=None, help="CSV file with distances per object per frame (optional)")
    p.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    args = p.parse_args()

    pipeline = DistanceEstimationPipeline(
        detector_weights=args.yolo_weights,
        depth_model_type=args.depth_type,
        depth_checkpoint=args.depth_checkpoint,
        device=args.device
    )
    pipeline.run(args.input, args.out_video, args.out_csv, conf=args.conf)


if __name__ == "__main__":
    main()
