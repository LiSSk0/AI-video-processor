import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from transformers import pipeline
from sys import exit

if not torch.cuda.is_available():
    print("CUDA is not available.")
    exit()

# Инициализируем пайплайн глобально при импорте модуля
try:
    depth_pipeline = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0  # gpu
    )
    print("[Depth Anything V2] Модель успешно загружена.")
except Exception as e:
    print(f"[Depth Anything V2] Ошибка при загрузке модели: {e}")
    depth_pipeline = None


def depth_map(video_path: str) -> (gr.update, str):
    if not video_path:
        raise gr.Error("Видео не загружено.")

    if depth_pipeline is None:
        raise gr.Error("Модель оценки глубины не инициализирована. Проверьте консоль.")

    # Открываем исходный видеофайл
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Создаем папку для результатов, если её нет
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "output_depth_anything_v2.mp4")

    # Настройка кодека для сохранения видео (.mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[Depth Anything V2] Началась обработка видео: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Конвертируем BGR (OpenCV) в RGB (для PIL и Hugging Face Transformers)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Инференс модели
        result = depth_pipeline(pil_img)
        depth_np = np.array(result["depth"])

        # Возвращаем карту глубин к исходному разрешению видео
        depth_resized = cv2.resize(depth_np, (width, height))

        # Применяем цветовую палитру Inferno (эффектный тепловой градиент)
        # Если нужна строго ч/б карта, замените на: color_depth = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
        color_depth = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)

        # Записываем обработанный кадр
        out.write(color_depth)

    # Закрываем потоки
    cap.release()
    out.release()
    print(f"[Depth Anything V2] Обработка завершена. Файл сохранен: {output_video_path}")

    # Возвращаем результаты строго под outputs=[mask_dropdown, output_video] в app.py:
    # 1. Скрываем дропдаун выбора слоев маски (gr.update(visible=False))
    # 2. Передаем путь к новому видео в плеер результатов
    return gr.update(visible=False), output_video_path
