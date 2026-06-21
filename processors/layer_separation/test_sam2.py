import cv2
from sam2_segmenter import SAM2Segmenter


CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"


segmenter = SAM2Segmenter(CHECKPOINT)


video_path = r"C:\Users\dasha\Downloads\Telegram Desktop\dynamic_backgraund.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

cap.release()

if not ret:
    raise Exception("Не удалось прочитать кадр")

masks = segmenter.segment(frame)

print("Найдено масок:", len(masks))


# сохраним первую маску
import cv2
import numpy as np

mask = masks[0].astype(np.uint8) * 255
cv2.imwrite("mask0.png", mask)