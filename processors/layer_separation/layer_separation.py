import os
import cv2
import numpy as np
import time
import shutil
from pathlib import Path
from processors.layer_separation.sam2_segmenter import SAM2Segmenter
from processors.layer_separation.config import CHECKPOINT_PATH, OUTPUT_DIR


class LayerSeparationProcessor:
    def __init__(self):
        self.segmenter = SAM2Segmenter(str(CHECKPOINT_PATH))
        self.chunk_size = 70

    def process(self, video_path: str, clicked_points: list) -> list[str]:
        start_time = time.time()

        cap, meta = self._extract_video_metadata(video_path)
        temp_chunk_dir = os.path.join(OUTPUT_DIR, f"temp_chunk_{meta['name']}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        object_points = [{"obj_id": 1, "point": clicked_points}]
        video_writers = []
        output_paths = []

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            for chunk_start_idx in range(0, meta["total_frames"], self.chunk_size):
                chunk_end_idx = min(chunk_start_idx + self.chunk_size, meta["total_frames"])
                current_chunk_len = chunk_end_idx - chunk_start_idx
                print(f"\n--- Обработка чанка кадров: {chunk_start_idx} - {chunk_end_idx} ---")

                chunk_frames = self._prepare_chunk_frames(cap, temp_chunk_dir, current_chunk_len)

                video_segments, inference_state = self.segmenter.process_video_tracking(
                    temp_chunk_dir,
                    object_points=object_points
                )

                last_frame_masks = self._write_layer_frames(
                    video_segments, chunk_frames, current_chunk_len, meta, video_writers, output_paths
                )

                object_points = self._calculate_next_centroids(last_frame_masks, object_points)

                self.segmenter.predictor.reset_state(inference_state)
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

    def _extract_video_metadata(self, video_path: str) -> tuple[cv2.VideoCapture, dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео по пути: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0
            while cap.isOpened():
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": total_frames,
            "name": Path(video_path).stem,
            "ext": Path(video_path).suffix
        }
        return cap, meta

    def _prepare_chunk_frames(self, cap: cv2.VideoCapture, temp_dir: str, chunk_len: int) -> list:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        chunk_frames = []
        for i in range(chunk_len):
            ret, frame = cap.read()
            if not ret:
                break
            chunk_frames.append(frame)
            frame_name = f"{i:05d}.jpg"
            cv2.imwrite(os.path.join(temp_dir, frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return chunk_frames


    def _write_layer_frames(self, video_segments, chunk_frames, chunk_len, meta, video_writers, output_paths) -> dict:
        last_frame_masks = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in video_segments:
            num_masks = len(out_obj_ids)

            while len(video_writers) < num_masks:
                layer_idx = len(video_writers) + 1
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out_path = os.path.join(OUTPUT_DIR, f"{meta['name']}_layer_{layer_idx}{meta['ext']}")
                writer = cv2.VideoWriter(out_path, fourcc, meta["fps"], (meta["width"], meta["height"]))
                video_writers.append(writer)
                output_paths.append(out_path)

            orig_frame = chunk_frames[out_frame_idx] if out_frame_idx < len(chunk_frames) else None

            for i in range(num_masks):
                mask_logits = out_mask_logits[i][0].cpu().numpy()
                mask_binary = (mask_logits > 0.0).astype(np.uint8) * 255

                if out_frame_idx == chunk_len - 1:
                    last_frame_masks[out_obj_ids[i]] = mask_logits > 0.0

                if orig_frame is not None:
                    layer_frame = cv2.bitwise_and(orig_frame, orig_frame, mask=mask_binary)
                    video_writers[i].write(layer_frame)

        return last_frame_masks

    def _calculate_next_centroids(self, last_frame_masks: dict, current_points: list) -> list:
        next_object_points = []
        for obj in current_points:
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
        return next_object_points