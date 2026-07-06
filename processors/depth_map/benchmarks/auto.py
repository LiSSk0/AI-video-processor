import os
import time
import csv
import torch
import psutil
import logging
from config.config_settings import OUTPUT_DIR, AUTO_FETCH_DIR
from processors.depth_map.depth_map import DepthMapProcessor

logger = logging.getLogger("Auto Fetch")


def main():
    test_dir = AUTO_FETCH_DIR

    video_files = [
        "428x240.mp4",
        "428x240.mp4",
        "640x360.mp4",
        "854x480.mp4",
        "1280x720.mp4",
        "1920x1080.mp4"
    ]

    logger.info("Loading model...")
    processor = DepthMapProcessor()
    logger.info("Model loaded.\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = []

    for video_file in video_files:
        video_path = os.path.join(test_dir, video_file)
        if not os.path.exists(video_path):
            logger.error(f"File not found: {video_path}")
            continue

        logger.info(f"Processing: {video_file}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        _, _ = processor.process(video_path)

        elapsed_time = time.time() - start_time

        process = psutil.Process()
        ram_usage = process.memory_info().rss / (1024 ** 2)

        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results.append({
            "name": video_file,
            "time": elapsed_time,
            "ram": ram_usage,
            "vram": vram_usage
        })

        logger.info(f"  Time: {elapsed_time:.2f}s, RAM: {ram_usage:.2f} MB, VRAM: {vram_usage:.2f} MB\n")

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_resolutions.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "time", "ram", "vram"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\nResults:")
    print(f"{'Video':<15} {'Time (s)':<12} {'RAM (MB)':<12} {'VRAM (MB)':<12}")
    for r in results:
        print(f"{r['name']:<15} {r['time']:<12.2f} {r['ram']:<12.2f} {r['vram']:<12.2f}")

    logger.info(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
