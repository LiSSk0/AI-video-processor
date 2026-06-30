import os
import cv2
import numpy as np
import time
from pathlib import Path
from config.config_settings import OUTPUT_DIR, SAM2_CHECKPOINT, DEVICE
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter


class OSTrackSeparationProcessor:
    def __init__(self):
        # Используем SAM2 для маски на первом кадре, а OSTrack (OpenCV Vit Tracker) для слежения за BBox
        self.sam2_segmenter = SAM2Segmenter(str(SAM2_CHECKPOINT))
        self.device = DEVICE

    def process(self, video_path: str, clicked_points: list) -> list[str]:
        start_time = time.time()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        name = Path(video_path).stem
        ext = ".mp4"

        out_background = os.path.join(OUTPUT_DIR, f"{name}_ostrack_background{ext}")
        out_object = os.path.join(OUTPUT_DIR, f"{name}_ostrack_object{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_bg = cv2.VideoWriter(out_background, fourcc, fps, (width, height))
        writer_obj = cv2.VideoWriter(out_object, fourcc, fps, (width, height))

        ret, first_frame = cap.read()
        if not ret:
            return []

        # 1. Получаем начальную маску от SAM2 по точкам клика
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        initial_mask = self.sam2_segmenter.get_image_mask(rgb_frame, clicked_points)

        y_idx, x_idx = np.where(initial_mask)
        if len(x_idx) == 0:
            print("[ERROR] OSTrack: Объект не найден на первом кадре.")
            return []

        # 2. Вычисляем Bounding Box для OSTrack
        x1, y1 = int(np.min(x_idx)), int(np.min(y_idx))
        x2, y2 = int(np.max(x_idx)), int(np.max(y_idx))
        bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)

        # Инициализируем OpenCV VIT трекер (в OpenCV 4.7.0+ доступен TrackerVIT, построенный на базе идей OSTrack)
        # Если TrackerVIT недоступен, используется надежный TrackerMIL/DaSiamRPN
        try:
            tracker = cv2.TrackerVIT_create()
            tracker.init(first_frame, bbox)
            print("[OSTrack] Успешно инициализирован TrackerVIT.")
        except AttributeError:
            tracker = cv2.TrackerMIL_create()
            tracker.init(first_frame, bbox)
            print("[OSTrack] TrackerVIT недоступен. Инициализирован TrackerMIL.")

        # Записываем первый кадр
        self._write_layers(first_frame, bbox, writer_obj, writer_bg, width, height)

        # 3. Цикл по остальным кадрам
        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обновляем трекер
            success, box = tracker.update(frame)

            if success:
                bbox = tuple(map(int, box))
            else:
                # Если потеряли объект, оставляем прошлый bbox
                pass

            self._write_layers(frame, bbox, writer_obj, writer_bg, width, height)
            frame_idx += 1

        cap.release()
        writer_bg.release()
        writer_obj.release()

        print(f"[INFO] Обработка OSTrack завершена за {time.time() - start_time:.2f} сек.")
        return [out_object, out_background]

    def _write_layers(self, frame, bbox, writer_obj, writer_bg, w, h):
        x, y, bw, bh = bbox

        # Создаем маску на основе Bounding Box
        mask_box = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_box, (x, y), (x + bw, y + bh), 255, -1)

        obj_layer = cv2.bitwise_and(frame, frame, mask=mask_box)
        bg_layer = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_box))

        writer_obj.write(obj_layer)
        writer_bg.write(bg_layer)