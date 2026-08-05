import os
import sys
import time
import csv
import gc
import json
import logging
import traceback
import random
import datetime
from pathlib import Path
import argparse
import cv2
import torch
import psutil
import numpy as np

from config import config_settings
from processors.layer_separation.sam2_separation.layer_separation import LayerSeparationProcessor
from processors.layer_separation.tap_separation.tap_processor import TAPSeparationProcessor
from processors.layer_separation.bbox_methods.ostrack_processor import OSTrackSeparationProcessor
from processors.layer_separation.bbox_methods.nanotrack_processor import NanoTrackSeparationProcessor
from processors.layer_separation.xmem_separation.xmem_processor import XMemSeparationProcessor
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter

logger = logging.getLogger("Benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = SCRIPT_DIR / "videos"
BASE_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR))

METHODS = [
    ("SAM2", LayerSeparationProcessor, 100),
    ("TAP", TAPSeparationProcessor, 30),
    ("OSTrack", OSTrackSeparationProcessor, 100),
    ("NanoTrack", NanoTrackSeparationProcessor, 100),
    ("XMem", XMemSeparationProcessor, 100),
]


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    duration = frame_count / fps if fps > 0 else 0
    size_mb = video_path.stat().st_size / (1024 * 1024)
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "size_mb": size_mb,
    }


def load_points(video_path):
    json_path = video_path.parent / f"{video_path.stem}_points.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        pts = data.get("points", [])
        return pts if pts else None
    except:
        return None


def limit_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    sampled = points[::step][:max_points]
    if len(sampled) < max_points and len(points) > len(sampled):
        additional = random.sample(points, min(max_points - len(sampled), len(points)))
        sampled.extend(additional)
    return sampled


def get_memory_usage():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return ram_mb, vram_mb


def get_max_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def evaluate_segmentation_quality(orig_video_path, result_video_path):
    cap_orig = cv2.VideoCapture(str(orig_video_path))
    cap_res = cv2.VideoCapture(str(result_video_path))

    areas = []
    perimeters = []
    compactnesses = []
    flow_consistencies = []

    prev_gray_orig = None
    prev_mask = None

    while True:
        ret_o, frame_o = cap_orig.read()
        ret_r, frame_r = cap_res.read()

        if not ret_o or not ret_r:
            break

        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_r, 1, 255, cv2.THRESH_BINARY)

        area = int(np.sum(mask > 0))
        areas.append(area)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and area > 0:
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            perimeters.append(perimeter)

            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
            else:
                compactness = 0.0
            compactnesses.append(compactness)
        else:
            perimeters.append(0.0)
            compactnesses.append(0.0)

        gray_o = cv2.cvtColor(frame_o, cv2.COLOR_BGR2GRAY)
        if prev_gray_orig is not None and prev_mask is not None and area > 0 and np.sum(prev_mask > 0) > 0:
            flow = cv2.calcOpticalFlowFarneback(prev_gray_orig, gray_o, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            y_indices, x_indices = np.where(prev_mask > 0)

            if len(x_indices) > 0:
                flow_x = flow[y_indices, x_indices, 0]
                flow_y = flow[y_indices, x_indices, 1]

                new_x = np.round(x_indices + flow_x).astype(int)
                new_y = np.round(y_indices + flow_y).astype(int)

                h, w = mask.shape
                valid_idx = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)
                new_x = new_x[valid_idx]
                new_y = new_y[valid_idx]

                pred_mask = np.zeros_like(mask)
                pred_mask[new_y, new_x] = 255

                intersection = np.sum((pred_mask > 0) & (mask > 0))
                union = np.sum((pred_mask > 0) | (mask > 0))

                flow_iou = intersection / union if union > 0 else 0.0
                flow_consistencies.append(flow_iou)
            else:
                flow_consistencies.append(0.0)
        else:
            flow_consistencies.append(1.0)

        prev_gray_orig = gray_o.copy()
        prev_mask = mask.copy()

    cap_orig.release()
    cap_res.release()

    if not areas:
        return {
            "mean_compactness": 0.0, "max_area_drop": 0.0,
            "temporal_flicker": 0.0, "mean_flow_consistency": 0.0
        }

    areas = np.array(areas)
    compactnesses = np.array(compactnesses)
    flow_consistencies = np.array(flow_consistencies)

    if len(areas) > 1:
        area_diffs = np.diff(areas)

        denom = np.where(areas[:-1] == 0, 1, areas[:-1])
        relative_drops = -area_diffs / denom
        max_area_drop = float(np.max(relative_drops))

        perimeters = np.array(perimeters)
        denom_p = np.where(perimeters[:-1] == 0, 1, perimeters[:-1])
        temporal_flicker = float(np.mean(np.abs(np.diff(perimeters)) / denom_p))
    else:
        max_area_drop = 0.0
        temporal_flicker = 0.0

    return {
        "mean_compactness": float(np.mean(compactnesses)),
        "max_area_drop": max(0.0, max_area_drop),
        "temporal_flicker": temporal_flicker,
        "mean_flow_consistency": float(np.mean(flow_consistencies))
    }


def run_benchmark(output_root):
    if not VIDEO_DIR.exists():
        logger.error(f"Video directory not found: {VIDEO_DIR}")
        return

    video_output_dir = Path("output_results")
    video_output_dir.mkdir(exist_ok=True)

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_root / f"benchmark_results_{timestamp}.csv"

    fieldnames = [
        "video_name", "method",
        "video_width", "video_height", "video_fps", "video_frame_count",
        "video_duration_sec", "video_size_mb",
        "num_points_original", "num_points_used",
        "status", "error_message",
        "time_sec",
        "ram_before_mb", "ram_after_mb", "ram_delta_mb",
        "vram_before_mb", "vram_after_mb", "vram_delta_mb", "vram_peak_mb",
        "output_files_created",
        "mean_compactness", "max_area_drop", "temporal_flicker", "mean_flow_consistency"
    ]

    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = [f for f in VIDEO_DIR.iterdir() if f.suffix.lower() in video_extensions]
    valid_videos = []
    for v in video_files:
        pts = load_points(v)
        if pts is not None:
            valid_videos.append((v, pts))
        else:
            logger.warning(f"Skipping {v.name} – no points found.")

    if not valid_videos:
        logger.warning("No videos with valid point files found.")
        csv_file.close()
        return

    logger.info(f"Found {len(valid_videos)} video(s) with valid points.")

    logger.info("Initializing shared SAM2 segmenter...")
    shared_sam2 = SAM2Segmenter(str(config_settings.SAM2_CHECKPOINT))

    orig_output_dir = config_settings.OUTPUT_DIR
    orig_temp_dir = config_settings.TEMP_DIR
    config_settings.OUTPUT_DIR = video_output_dir
    config_settings.TEMP_DIR = video_output_dir / "temp"
    config_settings.TEMP_DIR.mkdir(exist_ok=True)

    try:
        for video_path, points in valid_videos:
            logger.info(f"\n=== Processing video: {video_path.name} ===")
            try:
                video_info = get_video_info(video_path)
            except Exception as e:
                logger.error(f"Error retrieving video info: {e}")
                continue

            num_original = len(points)

            min_allowed_points = min(max_pts for _, _, max_pts in METHODS)
            limited = limit_points(points, min_allowed_points)
            num_used = len(limited)

            for method_name, cls, _ in METHODS:
                logger.info(f"  Running method: {method_name}...")

                row = {
                    "video_name": video_path.name,
                    "method": method_name,
                    "video_width": video_info["width"],
                    "video_height": video_info["height"],
                    "video_fps": video_info["fps"],
                    "video_frame_count": video_info["frame_count"],
                    "video_duration_sec": video_info["duration_sec"],
                    "video_size_mb": video_info["size_mb"],
                    "num_points_original": num_original,
                    "num_points_used": num_used,
                    "status": "success",
                    "error_message": "",
                    "time_sec": 0.0,
                    "ram_before_mb": 0.0,
                    "ram_after_mb": 0.0,
                    "ram_delta_mb": 0.0,
                    "vram_before_mb": 0.0,
                    "vram_after_mb": 0.0,
                    "vram_delta_mb": 0.0,
                    "vram_peak_mb": 0.0,
                    "output_files_created": 0,
                    "mean_compactness": 0.0,
                    "max_area_drop": 0.0,
                    "temporal_flicker": 0.0,
                    "mean_flow_consistency": 0.0
                }

                processor = None
                try:
                    reset_vram_stats()
                    processor = cls(sam2_segmenter=shared_sam2)
                    ram_before, vram_before = get_memory_usage()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    result_paths = processor.process(
                        video_path=str(video_path),
                        clicked_points=limited
                    )

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    ram_after, vram_after = get_memory_usage()
                    vram_peak = get_max_vram_mb()

                    if not result_paths:
                        raise RuntimeError("Processor returned an empty file list.")

                    # Фильтруем только существующие и непустые файлы
                    valid_files = [p for p in result_paths if Path(p).exists() and Path(p).stat().st_size > 0]

                    if not valid_files:
                        raise RuntimeError("No valid output files found.")

                    row["time_sec"] = end_time - start_time
                    row["ram_before_mb"] = ram_before
                    row["ram_after_mb"] = ram_after
                    row["ram_delta_mb"] = ram_after - ram_before
                    row["vram_before_mb"] = vram_before
                    row["vram_after_mb"] = vram_after
                    row["vram_delta_mb"] = vram_after - vram_before
                    row["vram_peak_mb"] = vram_peak
                    row["output_files_created"] = len(valid_files)

                    if valid_files:
                        logger.info(f"    Evaluating segmentation quality for {method_name}...")
                        metrics = evaluate_segmentation_quality(video_path, valid_files[0])
                        row.update(metrics)

                    row["status"] = "success"
                    logger.info(
                        f"    Success: finished in {row['time_sec']:.2f} sec, files created: {len(valid_files)}")

                except Exception as e:
                    row["status"] = "fail"
                    row["error_message"] = traceback.format_exc()
                    logger.error(f"    Error processing with {method_name}: {e}")

                finally:
                    if processor is not None:
                        del processor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    writer.writerow(row)
                    csv_file.flush()

            logger.info(f"Finished processing video: {video_path.name}\n")

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user.")
    finally:
        config_settings.OUTPUT_DIR = orig_output_dir
        config_settings.TEMP_DIR = orig_temp_dir
        csv_file.close()
        logger.info(f"Results successfully saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("benchmark_results"),
                        help="Directory to save CSV reports")
    args = parser.parse_args()
    run_benchmark(args.output_root)


if __name__ == "__main__":
    main()