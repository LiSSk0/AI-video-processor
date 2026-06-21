import torch
import numpy as np
import sys
import os

from sam_facebook_repo.sam2.build_sam import build_sam2
from sam_facebook_repo.sam2.sam2_image_predictor import SAM2ImagePredictor

# Импортируем инструмент Hydra для ручной настройки путей поиска
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir

class SAM2Segmenter:

    def __init__(self, checkpoint_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Вот этой строки у тебя сейчас не хватает внутри метода:
        model_cfg = "sam2.1/sam2.1_hiera_t.yaml"

        # Вычисляем абсолютный путь к папке с конфигами внутри sam_facebook_repo
        current_dir = os.path.dirname(os.path.abspath(__file__))  # processors/layer_separation
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  # AI-video-processor
        config_dir = os.path.join(project_root, "sam_facebook_repo", "sam2", "configs")

        # Инициализируем Hydra вручную
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            # Теперь model_cfg объявлена выше и успешно передастся в функцию
            sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)

        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device

    def segment(self, frame):
        """
        frame: numpy array (BGR из OpenCV)
        return: list[np.ndarray] masks
        """

        # Добавляем .copy(), чтобы сделать массив непрерывным в памяти
        frame_rgb = frame[:, :, ::-1].copy()

        self.predictor.set_image(frame_rgb)

        # Автоматическая сегментация
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True
        )

        return masks