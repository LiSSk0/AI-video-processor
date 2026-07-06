import os
import cv2
import torch
import numpy as np
import time
import logging
from torchvision.transforms import ToTensor

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "external" / "XMem"))

from model.network import XMem
from inference.inference_core import InferenceCore
from config.config_settings import OUTPUT_DIR, SAM2_CHECKPOINT, XMEM_CHECKPOINT, DEVICE
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter

logger = logging.getLogger("XMemProcessor")


class XMemSeparationProcessor:
    def __init__(self):
        self.sam2_segmenter = SAM2Segmenter(str(SAM2_CHECKPOINT))
        self.device = DEVICE
        self.checkpoint_path = str(XMEM_CHECKPOINT)

        # Основной конфиг XMem для инференса
        self.config = {
            'key_dim': 64,
            'value_dim': 512,
            'hidden_dim': 64,
            'single_object': False,
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 5,
            'max_mid_term_frames': 10,
            'max_long_term_elements': 10000,
        }

        # Загрузка весов XMem
        logger.info(f"Loading XMem checkpoint from {self.checkpoint_path}")
        load_start = time.time()
        self.network = XMem(self.config, self.checkpoint_path, map_location=self.device)
        self.network.eval()
        self.network.to(self.device)
        logger.info(f"XMem model loaded in {time.time() - load_start:.2f} seconds")

        self.im_transform = ToTensor()

    @torch.no_grad()
    def process(self, video_path: str, clicked_points: list) -> list[str]:
        total_start = time.time()
        logger.info(f"Starting XMem processing for video: {video_path}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video source.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video info: {width}x{height}, {fps:.2f} FPS")

        name = Path(video_path).stem
        ext = ".mp4"

        out_background = os.path.join(OUTPUT_DIR, f"{name}_xmem_background{ext}")
        out_object = os.path.join(OUTPUT_DIR, f"{name}_xmem_object{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_bg = cv2.VideoWriter(out_background, fourcc, fps, (width, height))
        writer_obj = cv2.VideoWriter(out_object, fourcc, fps, (width, height))

        ret, first_frame = cap.read()
        if not ret:
            logger.error("Failed to read the first frame.")
            cap.release()
            return []

        # 1. Получение стартовой маски через SAM2
        logger.info("Obtaining initial mask from SAM2...")
        sam_start = time.time()
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        initial_mask = self.sam2_segmenter.get_image_mask(rgb_frame, clicked_points)
        logger.info(f"SAM2 mask obtained in {time.time() - sam_start:.2f} seconds")

        y_idx, _ = np.where(initial_mask)
        if len(y_idx) == 0:
            logger.error("SAM2 could not detect an object for XMem initialization.")
            cap.release()
            return []

        # 2. Инициализация ядра инференса XMem
        logger.info("Initializing XMem InferenceCore...")
        processor = InferenceCore(self.network, config=self.config)
        processor.set_all_labels([1])

        # Преобразование первого кадра и маски
        frame_tensor = self.im_transform(rgb_frame).to(self.device)
        mask_tensor = torch.from_numpy(initial_mask.astype(np.float32)).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0)

        # Первый шаг с маской
        logger.info("Processing first frame with mask (initialization)...")
        prediction = processor.step(frame_tensor, mask_tensor)
        binary_mask = (prediction[1] > 0.5).cpu().numpy().astype(np.uint8) * 255
        self._write_layers(first_frame, binary_mask, writer_obj, writer_bg)

        # 3. Цикл по остальным кадрам
        frame_count = 1
        logger.info("Starting tracking loop...")
        track_start = time.time()
        log_interval = max(1, int(fps * 5))  # логировать каждые 5 секунд видео

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.im_transform(frame_rgb).to(self.device)

            prediction = processor.step(frame_tensor)
            binary_mask = (prediction[1] > 0.5).cpu().numpy().astype(np.uint8) * 255

            self._write_layers(frame, binary_mask, writer_obj, writer_bg)
            frame_count += 1

            # Логирование прогресса
            if frame_count % log_interval == 0:
                elapsed = time.time() - track_start
                speed = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count} frames, speed: {speed:.1f} FPS")

        cap.release()
        writer_bg.release()
        writer_obj.release()

        total_time = time.time() - total_start
        track_time = time.time() - track_start
        avg_fps = frame_count / track_time if track_time > 0 else 0

        logger.info(f"Tracking loop finished: {frame_count} frames processed in {track_time:.2f} s, average {avg_fps:.1f} FPS")
        logger.info(f"Total processing time (including SAM2 init) : {total_time:.2f} seconds")
        logger.info(f"Results saved to: {out_object} and {out_background}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [out_object, out_background]

    def _write_layers(self, frame, mask, writer_obj, writer_bg):
        obj_layer = cv2.bitwise_and(frame, frame, mask=mask)
        bg_layer = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

        writer_obj.write(obj_layer)
        writer_bg.write(bg_layer)