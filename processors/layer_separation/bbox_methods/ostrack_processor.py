import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from config.config_settings import OUTPUT_DIR, SAM2_CHECKPOINT, DEVICE
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter

logger = logging.getLogger("OSTrackProcessor")


class OSTrackSeparationProcessor:
    def __init__(self, sam2_segmenter=None):
        if sam2_segmenter is not None:
            self.sam2_segmenter = sam2_segmenter
        else:
            self.sam2_segmenter = SAM2Segmenter(str(SAM2_CHECKPOINT))
        self.device = DEVICE

    def process(self, video_path: str, clicked_points: list) -> list[str]:
        start_time = time.time()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        name = Path(video_path).stem
        ext = ".mp4"

        out_background = os.path.join(OUTPUT_DIR, f"{name}_ostrack_background{ext}")
        out_object = os.path.join(OUTPUT_DIR, f"{name}_ostrack_object{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_bg = cv2.VideoWriter(out_background, fourcc, fps, (width, height))
        writer_obj = cv2.VideoWriter(out_object, fourcc, fps, (width, height))

        ret, first_frame = cap.read()
        if not ret:
            logger.error("Failed to read first frame.")
            cap.release()
            writer_bg.release()
            writer_obj.release()
            return []

        logger.info("Extracting initial object mask using SAM2.")
        mask = self.sam2_segmenter.get_image_mask(first_frame, clicked_points)
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0:
            logger.error("SAM2 could not find any object for tracking.")
            cap.release()
            writer_bg.release()
            writer_obj.release()
            return []

        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        try:
            tracker = cv2.TrackerVIT_create()
            tracker.init(first_frame, bbox)
            logger.info("TrackerVIT successfully initialized.")
        except AttributeError:
            tracker = cv2.TrackerMIL_create()
            tracker.init(first_frame, bbox)
            logger.warning("TrackerVIT is unavailable. TrackerMIL initialized as fallback.")

        self._write_layers(first_frame, bbox, writer_obj, writer_bg, width, height)

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            success, box = tracker.update(frame)

            if success:
                bbox = tuple(map(int, box))
            else:
                pass

            self._write_layers(frame, bbox, writer_obj, writer_bg, width, height)
            frame_idx += 1

        cap.release()
        writer_bg.release()
        writer_obj.release()

        logger.info(f"OSTrack processing completed in {time.time() - start_time:.2f} seconds.")
        return [out_object, out_background]

    def _write_layers(self, frame, bbox, writer_obj, writer_bg, w, h):
        x, y, bw, bh = bbox

        mask_box = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_box, (x, y), (x + bw, y + bh), 255, -1)

        obj_layer = cv2.bitwise_and(frame, frame, mask=mask_box)
        bg_layer = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_box))

        writer_obj.write(obj_layer)
        writer_bg.write(bg_layer)