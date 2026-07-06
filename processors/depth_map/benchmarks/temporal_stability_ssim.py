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


def calculate_temporal_stability(video_path, model_name):
    """
    One specific video is taken and each current frame is compared with the previous frames inside it.
    If there is a flicker of depth artifacts in the video, the SSIM between adjacent frames will drop.
    """
    cap = cv2.VideoCapture(video_path)

    ssim_scores = []
    frame_idx = 0

    print(f"Starting temporal stability check for: {model_name}...")

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Couldn't read the first frame from {video_path}")
        cap.release()
        return 0.0

    prev_gray = prev_frame[:, :, 0]

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = curr_frame[:, :, 0]

        score, _ = ssim(prev_gray, curr_gray, full=True)
        ssim_scores.append(score)

        prev_gray = curr_gray

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frame transitions...")

    cap.release()

    if not ssim_scores:
        print(f"No frames processed for {video_path}")
        return 0.0

    return np.mean(ssim_scores)


if __name__ == "__main__":
    for path in VIDEOS:
        if not os.path.exists(path):
            print(f"File is not found: {path}.")
            exit(1)

    results = {}
    for video in VIDEOS:
        filename = os.path.basename(video)
        stability_score = calculate_temporal_stability(video, filename)
        results[filename] = stability_score
        print(f"Temporal SSIM for '{filename}': {stability_score:.6f}\n")

    for name, score in results.items():
        print(f"{name}: {score:.6f}")
