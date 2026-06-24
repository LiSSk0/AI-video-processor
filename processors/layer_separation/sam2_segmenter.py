import torch
import os
import numpy as np
import cv2
import gc
import sam2

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir


class SAM2Segmenter:

    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
        self.checkpoint_path = checkpoint_path

        sam2_dir = os.path.dirname(os.path.abspath(sam2.__file__))
        config_dir = os.path.join(sam2_dir, "configs")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Инициализируем hydra один раз здесь для всех билдеров
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            # 1. Создаем видео-предиктор для трекинга чанков
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)

            # 2. Создаем базовую модель для генератора масок
            sam_model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)

        # Инициализируем генератор масок, пока hydra была активна
        self.mask_generator = SAM2AutomaticMaskGenerator(sam_model)

        # Сразу удаляем тяжелую базовую модель из памяти, она больше не нужна
        del sam_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_video_tracking(self, video_path: str, object_points: list = None):
        inference_state = self.predictor.init_state(video_path=video_path)
        self.predictor.reset_state(inference_state)

        if not object_points:
            object_points = [{"obj_id": 1, "point": [[300, 300]]}]

        # Добавляем объекты на 0-й кадр
        for obj in object_points:
            pts = obj["point"]

            if isinstance(pts, list) and len(pts) > 0 and not isinstance(pts[0], list):
                pts = [pts]

            labels = [1] * len(pts)

            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj["obj_id"],
                points=pts,
                labels=labels
            )

        video_segments = self.predictor.propagate_in_video(inference_state)
        return video_segments, inference_state