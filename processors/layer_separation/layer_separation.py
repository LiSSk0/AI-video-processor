import os
import cv2
import numpy as np
import time
import shutil
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

    output_dir = r"C:\Users\dasha\Desktop\unik\practice integral\AI-video-processor\processors\layer_separation\outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    # Читаем все кадры в память CPU (или временный массив), чтобы нарезать на чанки
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    print(f"\n[START] Всего кадров в видео: {total_frames}")

    CHUNK_SIZE = 70
    print(f"Видео будет обработано чанками по {CHUNK_SIZE} кадров.")

    video_writers = []
    output_paths = []
    start_time = time.time()

    temp_chunk_dir = os.path.join(output_dir, f"temp_chunk_{name}")

    try:
        for chunk_start_idx in range(0, total_frames, CHUNK_SIZE):
            chunk_end_idx = min(chunk_start_idx + CHUNK_SIZE, total_frames)
            print(f"\n--- Обработка чанка кадров: {chunk_start_idx} - {chunk_end_idx} ---")

            # Пересоздаем чистую временную папку для кадров текущего чанка
            if os.path.exists(temp_chunk_dir):
                shutil.rmtree(temp_chunk_dir)
            os.makedirs(temp_chunk_dir, exist_ok=True)

            # Сохраняем во временную папку только кадры текущего чанка
            for i, frame_global_idx in enumerate(range(chunk_start_idx, chunk_end_idx)):
                frame_name = f"{i:05d}.jpg"
                cv2.imwrite(os.path.join(temp_chunk_dir, frame_name), all_frames[frame_global_idx],
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # Запускаем трекинг для текущего чанка
            video_segments, inference_state = segmenter.process_video_tracking(temp_chunk_dir)

            # Переносим маски из чанка в VideoWriter
            for out_frame_idx, out_obj_ids, out_mask_logits in video_segments:
                num_masks = len(out_obj_ids)

                # Создаем VideoWriter только один раз при обнаружении первой маски первого чанка
                while len(video_writers) < num_masks:
                    mask_idx = len(video_writers) + 1
                    out_path = os.path.join(output_dir, f"{name}_mask_{mask_idx}{ext}")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')

                    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    video_writers.append(writer)
                    output_paths.append(out_path)
                    print(f"Создан поток записи для объекта №{mask_idx}")

                # Записываем кадр-маску в видео
                for i in range(len(video_writers)):
                    mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

                    if i < num_masks:
                        mask_logits = out_mask_logits[i][0].cpu().numpy()
                        mask_data = (mask_logits > 0.0).astype(np.uint8) * 255
                        mask_frame = cv2.merge([mask_data, mask_data, mask_data])

                    video_writers[i].write(mask_frame)

                global_frame_num = chunk_start_idx + out_frame_idx
                percent = (global_frame_num / total_frames) * 100
                print(
                    f"[Прогресс] Кадр {global_frame_num}/{total_frames} ({percent:.1f}%) | Объявлено масок: {num_masks}")

            # Обязательно очищаем контекст этого чанка перед следующим, чтобы сбросить VRAM
            segmenter.predictor.reset_state(inference_state)
            del inference_state
            del video_segments

    finally:
        for writer in video_writers:
            writer.release()

        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)
            print(f"\n[INFO] Временная папка чанков удалена.")

    total_duration = time.time() - start_time
    print(f"Обработка завершена! Затрачено времени: {total_duration:.1f} сек.")
    return output_paths