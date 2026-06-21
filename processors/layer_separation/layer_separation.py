import cv2
from processors.layer_separation.sam2_segmenter import SAM2Segmenter


def separation_layers(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_read_success, frame = cap.read()
    cap.release()
    if frame_read_success:
        cv2.imwrite("debug_frame.jpg", frame)
    return video_path