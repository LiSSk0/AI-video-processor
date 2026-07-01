import torch
import logging
from cotracker.predictor import CoTrackerPredictor
from config.config_settings import DEVICE

logger = logging.getLogger("CoTrackerWrapper")


class CoTrackerWrapper:
    def __init__(self, checkpoint_path: str):
        self.device = DEVICE
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"CoTrackerPredictor initialized on device: {self.device}")

    def track_chunk(self, video_tensor: torch.Tensor, queries: torch.Tensor):

        logger.info("Starting CoTracker point tracking execution.")
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video_tensor, queries=queries)
        return pred_tracks, pred_visibility
