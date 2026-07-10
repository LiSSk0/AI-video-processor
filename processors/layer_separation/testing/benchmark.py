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
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
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


def init_models():
    logger.info("Инициализация моделей...")
    start = time.time()
    models = {}

    shared_sam2 = SAM2Segmenter(str(config_settings.SAM2_CHECKPOINT))

    for name, cls, _ in METHODS:
        logger.info(f"  Загрузка {name}...")
        models[name] = cls(sam2_segmenter=shared_sam2)

    logger.info(f"Инициализация завершена за {time.time() - start:.2f} сек.")
    return models


def run_benchmark(output_root):
    if not VIDEO_DIR.exists():
        logger.error(f"Папка с видео не найдена: {VIDEO_DIR}")
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
        "output_files_created"
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
            logger.warning(f"Пропускаем {v.name} – нет точек.")

    if not valid_videos:
        logger.warning("Нет видео с точками.")
        csv_file.close()
        return

    logger.info(f"Найдено видео с точками: {len(valid_videos)}")

    models = init_models()

    orig_output_dir = config_settings.OUTPUT_DIR
    orig_temp_dir = config_settings.TEMP_DIR
    config_settings.OUTPUT_DIR = video_output_dir
    config_settings.TEMP_DIR = video_output_dir / "temp"
    config_settings.TEMP_DIR.mkdir(exist_ok=True)

    try:
        for video_path, points in valid_videos:
            logger.info(f"\n=== Обработка видео: {video_path.name} ===")
            try:
                video_info = get_video_info(video_path)
            except Exception as e:
                logger.error(f"Ошибка получения информации: {e}")
                continue

            num_original = len(points)

            for method_name, _, max_points in METHODS:
                logger.info(f"  Метод {method_name}...")
                processor = models[method_name]
                limited = limit_points(points, max_points)
                num_used = len(limited)

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
                }

                try:
                    reset_vram_stats()
                    ram_before, vram_before = get_memory_usage()
                    start_time = time.perf_counter()

                    result_paths = processor.process(
                        video_path=str(video_path),
                        clicked_points=limited
                    )

                    end_time = time.perf_counter()
                    ram_after, vram_after = get_memory_usage()
                    vram_peak = get_max_vram_mb()

                    if not result_paths:
                        raise RuntimeError("Метод вернул пустой список.")

                    kept = []
                    for f in video_output_dir.glob(f"*{video_path.stem}*{method_name}*.mp4"):
                        if "background" in f.name.lower():
                            try:
                                f.unlink()
                            except:
                                pass
                        else:
                            kept.append(str(f))

                    if not kept:
                        raise RuntimeError("Не найдено выходных файлов с объектом.")

                    valid_files = [p for p in kept if Path(p).exists() and Path(p).stat().st_size > 0]

                    row["time_sec"] = end_time - start_time
                    row["ram_before_mb"] = ram_before
                    row["ram_after_mb"] = ram_after
                    row["ram_delta_mb"] = ram_after - ram_before
                    row["vram_before_mb"] = vram_before
                    row["vram_after_mb"] = vram_after
                    row["vram_delta_mb"] = vram_after - vram_before
                    row["vram_peak_mb"] = vram_peak
                    row["output_files_created"] = len(valid_files)
                    row["status"] = "success"

                    logger.info(f"    Успешно за {row['time_sec']:.2f} сек, файлов: {len(valid_files)}")

                except Exception as e:
                    row["status"] = "fail"
                    row["error_message"] = traceback.format_exc()
                    logger.error(f"    Ошибка: {e}")

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    writer.writerow(row)
                    csv_file.flush()

            logger.info(f"Видео {video_path.name} обработано.\n")

    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
    finally:
        config_settings.OUTPUT_DIR = orig_output_dir
        config_settings.TEMP_DIR = orig_temp_dir
        csv_file.close()
        logger.info(f"Результаты сохранены в {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("benchmark_results"),
                        help="Папка для сохранения CSV-отчётов")
    args = parser.parse_args()
    run_benchmark(args.output_root)


if __name__ == "__main__":
    main()