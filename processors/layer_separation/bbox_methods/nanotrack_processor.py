import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

from config.config_settings import OUTPUT_DIR, NANOTRACK_BACKBONE, NANOTRACK_HEAD, SAM2_CHECKPOINT
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter

logger = logging.getLogger("NanoTrackProcessor")


class NanoTrackSeparationProcessor:
    def __init__(self):
        self.sam2_segmenter = SAM2Segmenter(str(SAM2_CHECKPOINT))

        self.backbone_path = str(NANOTRACK_BACKBONE)
        self.head_path = str(NANOTRACK_HEAD)

    def process(self, video_path: str, clicked_points: list) -> list[str]:
        start_time = time.time()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error("Failed to open video.")
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

        writer_bg = cv2.VideoWriter(out_background, fourcc, fps, (width, height))

        writer_obj = cv2.VideoWriter(out_object, fourcc, fps, (width, height))

        if not writer_bg.isOpened():
            logger.error("Failed to open VideoWriter for background.")
            raise RuntimeError("Failed to open VideoWriter for background.")

        if not writer_obj.isOpened():
            logger.error("Failed to open VideoWriter for object.")
            raise RuntimeError("Failed to open VideoWriter for object.")

        ret, first_frame = cap.read()

        if not ret:
            logger.error("Failed to read the first frame.")
            return []

        rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        initial_mask = self.sam2_segmenter.get_image_mask(
            rgb,
            clicked_points
        )

        y_idx, x_idx = np.where(initial_mask)

        if len(x_idx) == 0:
            logger.error("SAM2 failed to isolate the object.")
            return []

        x1 = int(np.min(x_idx))
        y1 = int(np.min(y_idx))
        x2 = int(np.max(x_idx))
        y2 = int(np.max(y_idx))

        bbox = (x1, y1, x2 - x1, y2 - y1)

        if not os.path.exists(self.backbone_path):
            logger.error(f"Backbone file not found: {self.backbone_path}")
            raise FileNotFoundError(self.backbone_path)

        if not os.path.exists(self.head_path):
            logger.error(f"Head file not found: {self.head_path}")
            raise FileNotFoundError(self.head_path)

        logger.info(f"Backbone: {self.backbone_path}")
        logger.info(f"Head: {self.head_path}")

        try:
            params = cv2.TrackerNano_Params()

            params.backbone = self.backbone_path
            params.neckhead = self.head_path

            tracker = cv2.TrackerNano_create(params)

            tracker.init(first_frame, bbox)

            logger.info("Tracker successfully initialized.")

        except (cv2.error, AttributeError) as e:

            logger.error(f"Error creating NanoTrack: {e}")
            logger.warning("Using TrackerMIL instead.")

            tracker = cv2.TrackerMIL_create()
            tracker.init(first_frame, bbox)

        self._write_layers(first_frame, bbox, writer_obj, writer_bg, width, height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            success, box = tracker.update(frame)
            if success:
                bbox = tuple(map(int, box))

            self._write_layers(frame, bbox, writer_obj, writer_bg, width, height)

        cap.release()

        writer_bg.release()
        writer_obj.release()

        logger.info(f"NanoTrack completed processing in {time.time() - start_time:.2f} seconds.")

        return [out_object, out_background]

    def _write_layers(self, frame, bbox, writer_obj, writer_bg, width, height):

        x, y, bw, bh = bbox

        x = max(0, x)
        y = max(0, y)

        bw = min(bw, width - x)
        bh = min(bh, height - y)

        mask = np.zeros((height, width), dtype=np.uint8)

        if bw > 0 and bh > 0:
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

        object_layer = cv2.bitwise_and(frame,  frame, mask=mask)

        background_layer = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

        writer_obj.write(object_layer)
        writer_bg.write(background_layer)
