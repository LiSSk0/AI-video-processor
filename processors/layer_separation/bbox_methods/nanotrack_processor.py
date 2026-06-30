import os
import cv2
import numpy as np
import time
from pathlib import Path

from config.config_settings import OUTPUT_DIR, NANOTRACK_BACKBONE,NANOTRACK_HEAD, SAM2_CHECKPOINT
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter


class NanoTrackSeparationProcessor:
    def __init__(self):
        # SAM2 используется только для получения маски первого кадра
        self.sam2_segmenter = SAM2Segmenter(str(SAM2_CHECKPOINT))

        self.backbone_path = str(NANOTRACK_BACKBONE)
        self.head_path = str(NANOTRACK_HEAD)

    def process(self, video_path: str, clicked_points: list) -> list[str]:
        start_time = time.time()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("[ERROR] Не удалось открыть видео.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        name = Path(video_path).stem
        ext = ".mp4"

        out_background = os.path.join(
            OUTPUT_DIR,
            f"{name}_nanotrack_background{ext}"
        )

        out_object = os.path.join(
            OUTPUT_DIR,
            f"{name}_nanotrack_object{ext}"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer_bg = cv2.VideoWriter(
            out_background,
            fourcc,
            fps,
            (width, height)
        )

        writer_obj = cv2.VideoWriter(
            out_object,
            fourcc,
            fps,
            (width, height)
        )

        if not writer_bg.isOpened():
            raise RuntimeError("Не удалось открыть VideoWriter для background.")

        if not writer_obj.isOpened():
            raise RuntimeError("Не удалось открыть VideoWriter для object.")

        ret, first_frame = cap.read()

        if not ret:
            print("[ERROR] Не удалось прочитать первый кадр.")
            return []

        # --------------------------------------------------------
        # SAM2 -> маска первого кадра
        # --------------------------------------------------------

        rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        initial_mask = self.sam2_segmenter.get_image_mask(
            rgb,
            clicked_points
        )

        y_idx, x_idx = np.where(initial_mask)

        if len(x_idx) == 0:
            print("[ERROR] SAM2 не смог выделить объект.")
            return []

        x1 = int(np.min(x_idx))
        y1 = int(np.min(y_idx))
        x2 = int(np.max(x_idx))
        y2 = int(np.max(y_idx))

        bbox = (
            x1,
            y1,
            x2 - x1,
            y2 - y1
        )

        # --------------------------------------------------------
        # NanoTrack
        # --------------------------------------------------------

        if not os.path.exists(self.backbone_path):
            raise FileNotFoundError(self.backbone_path)

        if not os.path.exists(self.head_path):
            raise FileNotFoundError(self.head_path)

        print("[NanoTrack] Backbone:", self.backbone_path)
        print("[NanoTrack] Head:", self.head_path)

        try:

            params = cv2.TrackerNano_Params()

            params.backbone = self.backbone_path
            params.neckhead = self.head_path

            tracker = cv2.TrackerNano_create(params)

            tracker.init(first_frame, bbox)

            print("[NanoTrack] Tracker успешно инициализирован.")

        except (cv2.error, AttributeError) as e:

            print("[NanoTrack] Ошибка создания NanoTrack:")
            print(e)

            print("[NanoTrack] Используется TrackerMIL.")

            tracker = cv2.TrackerMIL_create()
            tracker.init(first_frame, bbox)

        # --------------------------------------------------------
        # Первый кадр
        # --------------------------------------------------------

        self._write_layers(
            first_frame,
            bbox,
            writer_obj,
            writer_bg,
            width,
            height
        )

        # --------------------------------------------------------
        # Обработка видео
        # --------------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            success, box = tracker.update(frame)

            if success:
                bbox = tuple(map(int, box))

            self._write_layers(
                frame,
                bbox,
                writer_obj,
                writer_bg,
                width,
                height
            )

        cap.release()

        writer_bg.release()
        writer_obj.release()

        print(
            f"[INFO] NanoTrack завершил обработку за "
            f"{time.time() - start_time:.2f} сек."
        )

        return [
            out_object,
            out_background
        ]

    def _write_layers(
        self,
        frame,
        bbox,
        writer_obj,
        writer_bg,
        width,
        height
    ):

        x, y, bw, bh = bbox

        x = max(0, x)
        y = max(0, y)

        bw = min(bw, width - x)
        bh = min(bh, height - y)

        mask = np.zeros((height, width), dtype=np.uint8)

        if bw > 0 and bh > 0:
            cv2.rectangle(
                mask,
                (x, y),
                (x + bw, y + bh),
                255,
                -1
            )

        object_layer = cv2.bitwise_and(
            frame,
            frame,
            mask=mask
        )

        background_layer = cv2.bitwise_and(
            frame,
            frame,
            mask=cv2.bitwise_not(mask)
        )

        writer_obj.write(object_layer)
        writer_bg.write(background_layer)