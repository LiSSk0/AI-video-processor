import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from config.config_settings import OUTPUT_DIR

# Настраиваем простой вывод в консоль
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger("SSIM_Evaluator_Grayscale")

VIDEO_SMALL = os.path.join(OUTPUT_DIR, "dav2_small.mp4")
VIDEO_BASE = os.path.join(OUTPUT_DIR, "dav2_base.mp4")
VIDEO_LARGE = os.path.join(OUTPUT_DIR, "dav2_large.mp4")  # эталон


def calculate_ssim(target_video_path, reference_video_path, model_name):
    cap_target = cv2.VideoCapture(target_video_path)
    cap_ref = cv2.VideoCapture(reference_video_path)

    ssim_scores = []
    frame_idx = 0

    print(f"Начинаем подсчет SSIM для модели: {model_name}...")

    while True:
        ret_target, frame_target = cap_target.read()
        ret_ref, frame_ref = cap_ref.read()

        if not ret_target or not ret_ref:
            break

        # Забираем только один канал (так как в ЧБ видео R=G=B)
        gray_target = frame_target[:, :, 0]
        gray_ref = frame_ref[:, :, 0]

        # Защита от разницы в разрешении кадров
        if gray_target.shape != gray_ref.shape:
            gray_target = cv2.resize(gray_target, (gray_ref.shape[1], gray_ref.shape[0]))

        # Расчет структурного сходства (SSIM)
        score, _ = ssim(gray_target, gray_ref, full=True)
        ssim_scores.append(score)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Обработано {frame_idx} кадров...")

    cap_target.release()
    cap_ref.release()

    if not ssim_scores:
        print(f"Не удалось извлечь кадры из {target_video_path}")
        return 0.0

    return np.mean(ssim_scores)


if __name__ == "__main__":
    # Проверка наличия файлов перед стартом
    for path in [VIDEO_SMALL, VIDEO_BASE, VIDEO_LARGE]:
        if not os.path.exists(path):
            print(f"Файл не найден: {path}.")
            exit(1)

    # Сравниваем Small с Large
    ssim_small = calculate_ssim(VIDEO_SMALL, VIDEO_LARGE, "Small")

    # Сравниваем Base с Large
    ssim_base = calculate_ssim(VIDEO_BASE, VIDEO_LARGE, "Base")

    print("SSIM Small to Large:", ssim_small)
    print("SSIM Base to Large:", ssim_base)
