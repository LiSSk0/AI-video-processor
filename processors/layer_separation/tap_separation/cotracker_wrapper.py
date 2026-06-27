import torch
from cotracker.predictor import CoTrackerPredictor

class CoTrackerWrapper:
    def __init__(self, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Инициализируем модель с оффлайн/онлайн поддержкой
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def track_chunk(self, video_tensor: torch.Tensor, queries: torch.Tensor):
        """
        Отслеживает точки на чанке кадров.
        :param video_tensor: Тензор формы (1, T, 3, H, W), RGB, нормализован 0-255.
        :param queries: Тензор формы (1, N, 3), где 3 это (t, x, y). t всегда 0 для начала чанка.
        :return: pred_tracks (1, T, N, 2), pred_visibility (1, T, N)
        """
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(video_tensor, queries=queries)
        return pred_tracks, pred_visibility