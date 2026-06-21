import os
import cv2
import numpy as np
import time
from processors.layer_separation.sam2_segmenter import SAM2Segmenter


def separation_layers(video_path: str) -> str:
    CHECKPOINT = r"C:\Users\dasha\Desktop\unik\practice integral\AI-video-processor\processors\layer_separation\checkpoints\sam2.1_hiera_tiny.pt"
    segmenter = SAM2Segmenter(CHECKPOINT)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео по пути: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = r"C:\Users\dasha\Desktop\unik\practice integral\AI-video-processor\processors\layer_separation\outputs"
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_mask_output{ext}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n[START] Начинается обработка видео.")
    print(f"Всего кадров для обработки: {total_frames}\n")

    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break  # Конец видео

            frame_idx += 1
            masks = segmenter.segment(frame)

            # Создаем пустой черный кадр на случай, если нейросеть не найдет ни одной маски
            mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

            if len(masks) > 0:
                first_mask = masks[0].astype(np.uint8) * 255
                mask_frame = cv2.merge([first_mask, first_mask, first_mask])

            # Записываем кадр-маску в выходное видео
            out.write(mask_frame)

            percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            frame_duration = time.time() - frame_start
            print(
                f"[Прогресс] Кадр {frame_idx}/{total_frames} ({percent:.1f}%) | Время кадра: {frame_duration:.2f} сек. | Найдено масок: {len(masks)}")
    finally:
        cap.release()
        out.release()

    total_duration = time.time() - start_time
    print(f"\nОбработка завершена успешно!")
    print(f"Затрачено времени всего: {total_duration:.1f} сек. (около {total_duration / 60:.1f} мин.)")
    print(f"Видео сохранено в: {output_path}")

    return output_path