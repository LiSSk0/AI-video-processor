import os
import cv2
import torch
import numpy as np
import time
import logging
import gc
from torchvision.transforms import ToTensor
from torch.amp import autocast
from contextlib import nullcontext

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "external" / "XMem"))

from model.network import XMem
from inference.inference_core import InferenceCore
from config.config_settings import OUTPUT_DIR, SAM2_CHECKPOINT, XMEM_CHECKPOINT, DEVICE
from processors.layer_separation.sam2_separation.sam2_segmenter import SAM2Segmenter

torch.backends.cudnn.benchmark = True

logger = logging.getLogger("XMemProcessor")


class XMemSeparationProcessor:
    def __init__(self, target_size: int = 360, sam2_segmenter=None):
        self.target_size = target_size

        if sam2_segmenter is not None:
            self.sam2_segmenter = sam2_segmenter
        else:
            self.sam2_segmenter = SAM2Segmenter(str(SAM2_CHECKPOINT))

        self.device = torch.device(DEVICE) if isinstance(DEVICE, str) else DEVICE
        self.checkpoint_path = str(XMEM_CHECKPOINT)

        self.config = {
            'key_dim': 64,  # Размерность ключей для поиска в памяти
            'value_dim': 512,  # Размерность значений (признаков объекта)
            'hidden_dim': 64,  # Размерность скрытых слоёв сети памяти
            'single_object': False,  # Режим множественных объектов (не один)
            'top_k': 20,  # Извлекать топ-20 похожих элементов из памяти
            'mem_every': 20,  # Добавлять кадр в память каждые 20 кадров
            'deep_update_every': -1,  # Глубокое обновление памяти отключено (-1)
            'enable_long_term': True,  # Включить долгосрочную память
            'enable_long_term_count_usage': True,  # Учитывать частоту использования элементов
            'num_prototypes': 24,  # Количество прототипов в долгосрочной памяти
            'min_mid_term_frames': 10,  # Минимум кадров в среднесрочной памяти
            'max_mid_term_frames': 15,  # Максимум кадров в среднесрочной памяти
            'max_long_term_elements': 1500,  # Максимум элементов в долгосрочной памяти
        }

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

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Original video: {original_width}x{original_height}, {original_fps:.2f} FPS")

        scale = self.target_size / min(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        new_width = (new_width // 16) * 16
        new_height = (new_height // 16) * 16
        logger.info(f"Resized to: {new_width}x{new_height} (target short side = {self.target_size})")

        name = Path(video_path).stem
        ext = ".mp4"

        out_background = os.path.join(OUTPUT_DIR, f"{name}_xmem_background{ext}")
        out_object = os.path.join(OUTPUT_DIR, f"{name}_xmem_object{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_bg = cv2.VideoWriter(out_background, fourcc, original_fps, (original_width, original_height))
        writer_obj = cv2.VideoWriter(out_object, fourcc, original_fps, (original_width, original_height))

        ret, first_frame = cap.read()
        if not ret:
            logger.error("Failed to read the first frame.")
            cap.release()
            return []

        logger.info("Obtaining initial mask from SAM2...")
        sam_start = time.time()
        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        initial_mask = self.sam2_segmenter.get_image_mask(rgb_frame, clicked_points)
        logger.info(f"SAM2 mask obtained in {time.time() - sam_start:.2f} seconds")

        if np.sum(initial_mask) == 0:
            logger.error("SAM2 could not detect an object for XMem initialization.")
            cap.release()
            return []

        logger.info("Initializing XMem InferenceCore...")
        processor = InferenceCore(self.network, config=self.config)
        processor.set_all_labels([1])

        first_frame_resized = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        frame_tensor = self.im_transform(first_frame_resized).to(self.device)

        mask_resized = cv2.resize(initial_mask.astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).to(self.device).unsqueeze(0)

        logger.info("Processing first frame with mask (initialization)...")
        use_amp = (self.device.type == 'cuda')
        with autocast(device_type='cuda', enabled=use_amp) if use_amp else nullcontext():
            prediction = processor.step(frame_tensor, mask_tensor)

        binary_mask = (prediction[1] > 0.5).cpu().numpy().astype(np.uint8) * 255
        binary_mask_full = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        self._write_layers(first_frame, binary_mask_full, writer_obj, writer_bg)

        frame_count = 1
        logger.info("Starting tracking loop...")
        track_start = time.time()
        log_interval = max(1, int(original_fps * 5))
        time_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            frame_tensor = self.im_transform(frame_resized).to(self.device)

            step_start = time.time()
            with autocast(device_type='cuda', enabled=use_amp) if use_amp else nullcontext():
                prediction = processor.step(frame_tensor)
            step_time = time.time() - step_start
            time_per_frame.append(step_time)

            binary_mask = (prediction[1] > 0.5).cpu().numpy().astype(np.uint8) * 255
            binary_mask_full = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            self._write_layers(frame, binary_mask_full, writer_obj, writer_bg)

            frame_count += 1

            if frame_count % log_interval == 0:
                avg_time = np.mean(time_per_frame[-log_interval:]) if time_per_frame else 0
                elapsed = time.time() - track_start
                speed = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count} frames, avg time/frame: {avg_time:.3f} s, speed: {speed:.1f} FPS")

            if frame_count % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                work_size = processor.memory.work_mem.size if hasattr(processor.memory, 'work_mem') else 0
                long_size = processor.memory.long_mem.size if hasattr(processor.memory, 'long_mem') else 0
                logger.info(f"Memory: working={work_size} elements, long-term={long_size} elements")

        cap.release()
        writer_bg.release()
        writer_obj.release()

        total_time = time.time() - total_start
        track_time = time.time() - track_start
        avg_fps = frame_count / track_time if track_time > 0 else 0
        avg_time_per_frame = np.mean(time_per_frame) if time_per_frame else 0

        logger.info(f"Tracking loop finished: {frame_count} frames processed in {track_time:.2f} s, average {avg_fps:.1f} FPS, avg time/frame {avg_time_per_frame:.3f} s")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {out_object} and {out_background}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return [out_object, out_background]

    def _write_layers(self, frame, mask, writer_obj, writer_bg):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # если почему-то 3-канальная, берём первый

        obj_layer = cv2.bitwise_and(frame, frame, mask=mask)
        bg_mask = cv2.bitwise_not(mask)
        bg_layer = cv2.bitwise_and(frame, frame, mask=bg_mask)

        writer_obj.write(obj_layer)
        writer_bg.write(bg_layer)