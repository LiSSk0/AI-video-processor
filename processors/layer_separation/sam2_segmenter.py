import torch
import os

# Импортируем специализированный билдер для видео-предиктора
from sam_facebook_repo.sam2.build_sam import build_sam2_video_predictor

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir


class SAM2Segmenter:

    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_cfg = "sam2.1/sam2.1_hiera_t.yaml"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        config_dir = os.path.join(project_root, "sam_facebook_repo", "sam2", "configs")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)

    def process_video_tracking(self, video_path: str):
        inference_state = self.predictor.init_state(video_path=video_path)

        self.predictor.reset_state(inference_state)

        # здесь точка-заглушка, потом нужно передавать координаты интересующего объекта
        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=[[300, 300]],
            labels=[1]
        )

        video_segments = self.predictor.propagate_in_video(inference_state)
        return video_segments, inference_state