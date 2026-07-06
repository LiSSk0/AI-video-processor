import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from config.config_settings import OUTPUT_DIR

STANDARD = os.path.join(OUTPUT_DIR, "output_DAv2_3840x2160.mp4")  # эталон
VIDEOS = [os.path.join(OUTPUT_DIR, "output_DAv2_428x240.mp4"),
          os.path.join(OUTPUT_DIR, "output_DAv2_640x360.mp4"),
          os.path.join(OUTPUT_DIR, "output_DAv2_854x480.mp4"),
          os.path.join(OUTPUT_DIR, "output_DAv2_1280x720.mp4"),
          os.path.join(OUTPUT_DIR, "output_DAv2_1920x1080.mp4")]


def calculate_ssim(target_video_path, reference_video_path, model_name):
    cap_target = cv2.VideoCapture(target_video_path)
    cap_ref = cv2.VideoCapture(reference_video_path)

    ssim_scores = []
    frame_idx = 0

    print(f"Starting SSIM for model: {model_name}...")

    while True:
        ret_target, frame_target = cap_target.read()
        ret_ref, frame_ref = cap_ref.read()

        if not ret_target or not ret_ref:
            break

        gray_target = frame_target[:, :, 0]
        gray_ref = frame_ref[:, :, 0]

        if gray_target.shape != gray_ref.shape:
            gray_ref = cv2.resize(gray_ref, (gray_target.shape[1], gray_target.shape[0]))

        score, _ = ssim(gray_target, gray_ref, full=True)
        ssim_scores.append(score)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap_target.release()
    cap_ref.release()

    if not ssim_scores:
        print(f"Couldn't extract frames from {target_video_path}")
        return 0.0

    return np.mean(ssim_scores)


if __name__ == "__main__":
    for path in VIDEOS:
        if not os.path.exists(path):
            print(f"File is not found: {path}.")
            exit(1)

    for video in VIDEOS:
        filename = os.path.basename(video)
        ssim_result = calculate_ssim(video, STANDARD, filename)
        print(f"SSIM '{filename}' to '{os.path.basename(STANDARD)}': ", ssim_result)
