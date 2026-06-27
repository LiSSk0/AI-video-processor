import torch
import os
import gc
import sam2
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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

        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)
            sam_model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)

        self.mask_generator = SAM2AutomaticMaskGenerator(sam_model)

        del sam_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_video_tracking(self, video_path: str, object_points: list = None):
        inference_state = self.predictor.init_state(video_path=video_path)
        self.predictor.reset_state(inference_state)

        if not object_points:
            object_points = [{"obj_id": 1, "point": [[300, 300]]}]

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

    def get_image_mask(self, image: np.ndarray, point_coords: list) -> np.ndarray:
        """
        Получает маску для одного изображения (первого кадра) по кликнутой точке.
        """
        import os
        import sam2
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_dir

        # --- Добавляем инициализацию Hydra ---
        sam2_dir = os.path.dirname(os.path.abspath(sam2.__file__))
        config_dir = os.path.join(sam2_dir, "configs")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            sam_model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)
        # -------------------------------------

        image_predictor = SAM2ImagePredictor(sam_model)
        image_predictor.set_image(image)

        pts = np.array(point_coords, dtype=np.float32)
        labels = np.ones(len(pts), dtype=np.int32)

        masks, scores, _ = image_predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=False
        )

        del image_predictor
        del sam_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return masks[0] > 0.0


