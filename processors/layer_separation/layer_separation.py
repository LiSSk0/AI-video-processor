import os
import cv2
import numpy as np
import time
from processors.layer_separation.sam2_segmenter import SAM2Segmenter


def separation_layers(video_path: str) -> list[str]:
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
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    # Список для хранения объектов VideoWriter и путей к файлам
    video_writers = []
    output_paths = []

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

            num_masks = len(masks)

            # Динамически создаем VideoWriter для новых масок, если их больше, чем было раньше
            while len(video_writers) < num_masks:
                mask_idx = len(video_writers) + 1
                out_path = os.path.join(output_dir, f"{name}_mask_{mask_idx}{ext}")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')

                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                video_writers.append(writer)
                output_paths.append(out_path)
                print(f"[INFO] Создан новый поток записи для маски №{mask_idx}")

            # Записываем каждую маску в свой файл
            for i in range(len(video_writers)):
                mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Если для этого индекса в текущем кадре есть маска — рисуем её
                if i < num_masks:
                    mask_data = masks[i].astype(np.uint8) * 255
                    mask_frame = cv2.merge([mask_data, mask_data, mask_data])

                video_writers[i].write(mask_frame)

            percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            frame_duration = time.time() - frame_start
            print(
                f"[Прогресс] Кадр {frame_idx}/{total_frames} ({percent:.1f}%) | Время: {frame_duration:.2f} сек. | Найдено масок: {num_masks}")

    finally:
        cap.release()
        for writer in video_writers:
            writer.release()

    total_duration = time.time() - start_time
    print(f"\nОбработка завершена успешно!")
    print(f"Затрачено времени всего: {total_duration:.1f} сек.")
    print(f"Сохранено слоев-масок: {len(output_paths)}")

    # Если масок вообще не было найдено, вернем пустой список
    return output_paths