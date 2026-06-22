import torch
import os
import numpy as np
import cv2
import gc  # Импортируем сборщик мусора

# Импортируем оба билдера
from sam_facebook_repo.sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam_facebook_repo.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir


class SAM2Segmenter:

    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
        self.checkpoint_path = checkpoint_path

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        config_dir = os.path.join(project_root, "sam_facebook_repo", "sam2", "configs")

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

    def get_auto_points_for_first_frame(self, first_frame_path: str):
        """
        Анализирует первый кадр, находит уникальные объекты
        и возвращает центральные точки для каждого объекта.
        """
        # Если генератор масок уже был удален в предыдущем вызове
        if self.mask_generator is None:
            print("[WARN] Генератор масок уже был уничтожен для экономии памяти.")
            return []

        image = cv2.imread(first_frame_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Выбираем безопасный тип данных в зависимости от поддержки bfloat16 видеокартой
        if self.device == "cuda":
            # Проверяем, поддерживает ли видеокарта современный bfloat16
            if torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.float32

        # Генерируем маски ВНУТРИ безопасного контекста autocast
        # Это решает проблему с CUBLAS_STATUS_EXECUTION_FAILED
        try:
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    masks = self.mask_generator.generate(image_rgb)
            else:
                masks = self.mask_generator.generate(image_rgb)
        except RuntimeError as e:
            # Если даже так упало, пробуем последний резервный вариант в Float16
            if "CUDA error" in str(e) and self.device == "cuda":
                print("[INFO] Сбой cuBLAS. Пробуем резервный запуск в Float16...")
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    masks = self.mask_generator.generate(image_rgb)
            else:
                raise e

        object_points = []
        for idx, mask_info in enumerate(masks):
            mask = mask_info['segmentation']
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0:
                continue

            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))

            if not mask[center_y, center_x]:
                center_x = int(x_indices[0])
                center_y = int(y_indices[0])

            object_points.append({
                "obj_id": idx + 1,
                "point": [center_x, center_y]
            })

        # === ЖЕСТКАЯ ОЧИСТКА ПАМЯТИ ===
        del self.mask_generator
        self.mask_generator = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return object_points

    def process_video_tracking(self, video_path: str, object_points: list = None):
        inference_state = self.predictor.init_state(video_path=video_path)
        self.predictor.reset_state(inference_state)

        if not object_points:
            object_points = [{"obj_id": 1, "point": [[300, 300]]}]

        # Добавляем объекты на 0-й кадр
        for obj in object_points:
            pts = obj["point"]

            # Если пришла одна точка [x, y], оборачиваем её в список [[x, y]]
            if isinstance(pts, list) and len(pts) > 0 and not isinstance(pts[0], list):
                pts = [pts]

            # Создаем массив лейблов «1» (foreground) точно такой же длины, как и массив точек
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