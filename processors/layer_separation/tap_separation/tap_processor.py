import os
import cv2
import numpy as np
import time
import torch
import gc
from pathlib import Path

from processors.layer_separation.config import CHECKPOINT_PATH, OUTPUT_DIR
from processors.layer_separation.sam2_segmenter import SAM2Segmenter
from processors.layer_separation.tap_separation.cotracker_wrapper import CoTrackerWrapper


class TAPSeparationProcessor:
    def __init__(self):
        # Указываем точный путь к скачанному scaled_offline.pth
        cotracker_ckpt = os.path.join(os.path.dirname(CHECKPOINT_PATH), "scaled_offline.pth")

        self.sam2_segmenter = SAM2Segmenter(str(CHECKPOINT_PATH))
        self.cotracker = CoTrackerWrapper(cotracker_ckpt)

        # Настройки разбиения и плотности сетки
        self.chunk_size = 60
        self.grid_step = 12
    def process(self, video_path: str, clicked_points: list) -> list[str]:
        start_time = time.time()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        name = Path(video_path).stem
        ext = Path(video_path).suffix

        out_masked = os.path.join(OUTPUT_DIR, f"{name}_tap_background{ext}")
        out_object = os.path.join(OUTPUT_DIR, f"{name}_tap_object{ext}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_bg = cv2.VideoWriter(out_masked, fourcc, fps, (width, height))
        writer_obj = cv2.VideoWriter(out_object, fourcc, fps, (width, height))

        # --- ЭТАП 1: Инициализация по первому кадру (SAM2) ---
        ret, first_frame = cap.read()
        if not ret:
            return []

        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        print("[TAP] Получение начальной маски объекта через SAM2...")

        # Исправленная строка (получаем маску только для изображения, а не для всего видео):
        initial_mask = self.sam2_segmenter.get_image_mask(rgb_frame, clicked_points)

        # --- ЭТАП 2: Выбор точек для трекинга ---
        y_idx, x_idx = np.where(initial_mask)
        if len(x_idx) == 0:
            print("[ERROR] SAM2 не нашел объект.")
            return []

        # Фильтруем (прореживаем) точки, чтобы не перегружать память
        pts_x = x_idx[::self.grid_step]
        pts_y = y_idx[::self.grid_step]
        current_points = np.column_stack((pts_x, pts_y)).astype(np.float32)
        print(f"[TAP] Найдено точек для отслеживания: {len(current_points)}")

        # --- ЭТАП 3 & 4: Динамический трекинг и Convex Hull (Чанками) ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for chunk_start in range(0, total_frames, self.chunk_size):
            chunk_frames_bgr = []
            chunk_frames_rgb_tensor = []

            for _ in range(self.chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames_bgr.append(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chunk_frames_rgb_tensor.append(torch.from_numpy(rgb).permute(2, 0, 1).float())

            if not chunk_frames_bgr:
                break

            T = len(chunk_frames_bgr)
            print(f"--- Обработка чанка: {chunk_start} - {chunk_start + T} ---")

            # Подготовка видео тензора [1, T, 3, H, W]
            video_tensor = torch.stack(chunk_frames_rgb_tensor).unsqueeze(0).to(self.cotracker.device)

            # Подготовка запросов: [1, N, 3] -> (t=0, x, y)
            N = len(current_points)
            queries = np.zeros((N, 3), dtype=np.float32)
            queries[:, 1:] = current_points
            queries_tensor = torch.from_numpy(queries).unsqueeze(0).to(self.cotracker.device)

            # Трекинг
            pred_tracks, pred_vis = self.cotracker.track_chunk(video_tensor, queries_tensor)

            # pred_tracks имеет форму (1, T, N, 2)
            tracks_np = pred_tracks[0].cpu().numpy()  # (T, N, 2)
            vis_np = pred_vis[0].cpu().numpy()  # (T, N)

            for t in range(T):
                frame = chunk_frames_bgr[t]
                valid_mask = vis_np[t] > 0.5
                valid_points = tracks_np[t][valid_mask]

                # Создание маски Convex Hull
                mask_hull = np.zeros((height, width), dtype=np.uint8)
                if len(valid_points) >= 3:
                    # Приводим к int32 для OpenCV
                    pts_int = valid_points.astype(np.int32)
                    hull = cv2.convexHull(pts_int)
                    cv2.fillConvexPoly(mask_hull, hull, 255)

                # Вырезание слоев
                obj_layer = cv2.bitwise_and(frame, frame, mask=mask_hull)
                bg_layer = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_hull))

                writer_obj.write(obj_layer)
                writer_bg.write(bg_layer)

            # Обновляем current_points для следующего чанка (последний кадр текущего чанка)
            # Берем только видимые точки на последнем кадре
            last_valid_mask = vis_np[-1] > 0.5
            current_points = tracks_np[-1][last_valid_mask]

            # Очистка памяти видеокарты
            del video_tensor
            del queries_tensor
            del pred_tracks
            torch.cuda.empty_cache()
            gc.collect()

        cap.release()
        writer_bg.release()
        writer_obj.release()

        total_duration = time.time() - start_time
        print(f"[INFO] Обработка TAP завершена за {total_duration:.2f} сек.")
        return [out_object]