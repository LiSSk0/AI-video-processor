import os
import cv2
import numpy as np
import time
import shutil
from processors.layer_separation.sam2_segmenter import SAM2Segmenter
import gc


def separation_layers(video_path: str, clicked_point: list) -> list[str]:
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

    # Если OpenCV не может отдать точное количество кадров, считаем вручную
    if total_frames <= 0:
        total_frames = 0
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            total_frames += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_dir = r"C:\Users\dasha\Desktop\unik\practice integral\AI-video-processor\processors\layer_separation\outputs"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    CHUNK_SIZE = 70
    video_writers = []
    output_paths = []
    start_time = time.time()

    temp_chunk_dir = os.path.join(output_dir, f"temp_chunk_{name}")

    # Инициализируем object_points точкой клика, переданной из Gradio
    object_points = [{"obj_id": 1, "point": clicked_point}]

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for chunk_start_idx in range(0, total_frames, CHUNK_SIZE):
            chunk_end_idx = min(chunk_start_idx + CHUNK_SIZE, total_frames)
            current_chunk_len = chunk_end_idx - chunk_start_idx
            print(f"\n--- Обработка чанка кадров: {chunk_start_idx} - {chunk_end_idx} ---")

            if os.path.exists(temp_chunk_dir):
                shutil.rmtree(temp_chunk_dir)
            os.makedirs(temp_chunk_dir, exist_ok=True)

            chunk_frames = []
            for i in range(current_chunk_len):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)
                frame_name = f"{i:05d}.jpg"
                cv2.imwrite(os.path.join(temp_chunk_dir, frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # Передаем object_points в метод сегментера
            video_segments, inference_state = segmenter.process_video_tracking(
                temp_chunk_dir,
                object_points=object_points
            )
            last_frame_masks = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in video_segments:
                num_masks = len(out_obj_ids)

                while len(video_writers) < num_masks:
                    layer_idx = len(video_writers) + 1
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out_path = os.path.join(output_dir, f"{name}_layer_{layer_idx}{ext}")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    video_writers.append(writer)
                    output_paths.append(out_path)

                orig_frame = chunk_frames[out_frame_idx] if out_frame_idx < len(chunk_frames) else None

                for i in range(num_masks):
                    mask_logits = out_mask_logits[i][0].cpu().numpy()
                    mask_binary = (mask_logits > 0.0).astype(np.uint8) * 255

                    if out_frame_idx == current_chunk_len - 1:
                        last_frame_masks[out_obj_ids[i]] = mask_logits > 0.0

                    if orig_frame is not None:
                        layer_frame = cv2.bitwise_and(orig_frame, orig_frame, mask=mask_binary)
                        video_writers[i].write(layer_frame)

            # Пересчет центра маски для следующего чанка
            next_object_points = []
            for obj in object_points:
                obj_id = obj["obj_id"]
                if obj_id in last_frame_masks:
                    mask = last_frame_masks[obj_id]
                    y_indices, x_indices = np.where(mask)

                    if len(x_indices) > 0:
                        center_x = int(np.mean(x_indices))
                        center_y = int(np.mean(y_indices))

                        if not mask[center_y, center_x]:
                            center_x = int(x_indices[0])
                            center_y = int(y_indices[0])

                        next_object_points.append({"obj_id": obj_id, "point": [center_x, center_y]})
                    else:
                        next_object_points.append(obj)
                else:
                    next_object_points.append(obj)

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
    print(f"[INFO] Обработка завершена за {total_duration:.2f} сек.")
    return output_paths