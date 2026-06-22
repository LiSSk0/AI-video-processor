import os
import cv2
import numpy as np
import time
import shutil
from processors.layer_separation.sam2_segmenter import SAM2Segmenter


def separation_layers(video_path: str) -> list[str]:
    CHECKPOINT = r"C:\Users\dasha\Desktop\unik\practice integral\AI-video-processor\processors\layer_separation\checkpoints\sam2.1_hiera_tiny.pt"
    segmenter = SAM2Segmenter(CHECKPOINT)

    # 1. Открываем видео для получения параметров
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео по пути: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Если OpenCV не может отдать точное количество кадров, считаем вручную быстро
    if total_frames <= 0:
        total_frames = 0
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            total_frames += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Сбрасываем позицию в начало

    print(f"\n[START] Всего кадров в видео: {total_frames}")

    output_dir = r"C:\Users\dasha\Desktop\unik\practice integral\AI-video-processor\processors\layer_separation\outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    CHUNK_SIZE = 70
    video_writers = []
    output_paths = []
    start_time = time.time()

    temp_chunk_dir = os.path.join(output_dir, f"temp_chunk_{name}")
    if os.path.exists(temp_chunk_dir):
        shutil.rmtree(temp_chunk_dir)
    os.makedirs(temp_chunk_dir, exist_ok=True)

    # ПЕРВЫЙ ПРОХОД: Сохраняем только самый первый кадр для анализа объектов
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Не удалось прочитать первый кадр видео.")

    first_frame_path = os.path.join(temp_chunk_dir, "00000.jpg")
    cv2.imwrite(first_frame_path, first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print("[INFO] Анализ первого кадра для поиска всех объектов...")
    object_points = segmenter.get_auto_points_for_first_frame(first_frame_path)
    print(f"[INFO] Найдено уникальных объектов: {len(object_points)}")

    if not object_points:
        print("[WARN] Объекты не найдены. Возвращаем пустой список.")
        cap.release()
        return []

    try:
        # Сбрасываем указатель видео в начало для покадрового чтения чанками
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for chunk_start_idx in range(0, total_frames, CHUNK_SIZE):
            chunk_end_idx = min(chunk_start_idx + CHUNK_SIZE, total_frames)
            current_chunk_len = chunk_end_idx - chunk_start_idx
            print(f"\n--- Обработка чанка кадров: {chunk_start_idx} - {chunk_end_idx} ---")

            # Очищаем и пересоздаем временную папку для текущего чанка
            if os.path.exists(temp_chunk_dir):
                shutil.rmtree(temp_chunk_dir)
            os.makedirs(temp_chunk_dir, exist_ok=True)

            # Нарезаем кадры ТЕКУЩЕГО чанка на диск, не забивая оперативную память arrays
            for i in range(current_chunk_len):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_name = f"{i:05d}.jpg"
                cv2.imwrite(os.path.join(temp_chunk_dir, frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # Передаем путь к чанку и найденные (или обновленные с прошлого чанка) точки объектов в трекер
            video_segments, inference_state = segmenter.process_video_tracking(temp_chunk_dir, object_points)

            # Переменная для сохранения масок самого последнего кадра в текущем чанке
            last_frame_masks = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in video_segments:
                num_masks = len(out_obj_ids)

                # Динамически создаем VideoWriter для каждого найденного ID объекта
                while len(video_writers) < num_masks:
                    mask_idx = len(video_writers) + 1
                    out_path = os.path.join(output_dir, f"{name}_mask_{mask_idx}{ext}")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    video_writers.append(writer)
                    output_paths.append(out_path)
                    print(f"Создан поток записи для объекта №{mask_idx}")

                # Записываем маску каждого объекта в свой файл
                for i in range(len(video_writers)):
                    mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

                    if i < num_masks:
                        mask_logits = out_mask_logits[i][0].cpu().numpy()
                        mask_data = (mask_logits > 0.0).astype(np.uint8) * 255
                        mask_frame = cv2.merge([mask_data, mask_data, mask_data])

                        # Сохраняем логику маски для вычисления точек на последнем кадре чанка
                        if out_frame_idx == current_chunk_len - 1:
                            last_frame_masks[out_obj_ids[i]] = mask_logits > 0.0

                    video_writers[i].write(mask_frame)

                global_frame_num = chunk_start_idx + out_frame_idx
                percent = (global_frame_num / total_frames) * 100
                if out_frame_idx % 10 == 0:
                    print(f"[Прогресс] Кадр {global_frame_num}/{total_frames} ({percent:.1f}%)")

            # === ОБНОВЛЕНИЕ ТОЧЕК ДЛЯ СЛЕДУЮЩЕГО ЧАНКА ===
            # Вычисляем центроиды масок на последнем кадре этого чанка, чтобы передать их дальше
            next_object_points = []
            for obj in object_points:
                obj_id = obj["obj_id"]
                # Если объект успешно оттрекался на последнем кадре текущего чанка
                if obj_id in last_frame_masks:
                    mask = last_frame_masks[obj_id]
                    y_indices, x_indices = np.where(mask)

                    if len(x_indices) > 0:
                        center_x = int(np.mean(x_indices))
                        center_y = int(np.mean(y_indices))

                        # Защита от выхода за пределы геометрии маски (берем первый попавшийся пиксель, если центр пустой)
                        if not mask[center_y, center_x]:
                            center_x = int(x_indices[0])
                            center_y = int(y_indices[0])

                        next_object_points.append({"obj_id": obj_id, "point": [center_x, center_y]})
                    else:
                        # Если маска исчезла в этом кадре, сохраняем старую точку как запасную
                        next_object_points.append(obj)
                else:
                    next_object_points.append(obj)

            # Перезаписываем точки новыми координатами для следующей итерации цикла
            object_points = next_object_points

            segmenter.predictor.reset_state(inference_state)
            del inference_state
            del video_segments

    finally:
        cap.release()
        for writer in video_writers:
            writer.release()

        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)
            print(f"\n[INFO] Временная папка чанков удалена.")

    total_duration = time.time() - start_time
    print(f"Обработка завершена! Затрачено времени: {total_duration:.1f} сек.")

