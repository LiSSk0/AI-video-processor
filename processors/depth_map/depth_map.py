import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from transformers import pipeline
from sys import exit
from enum import IntEnum
from config.config_settings import OUTPUT_DIR, DEPTH_MODEL_NAME
from config.config_settings import logger


class DeviceType(IntEnum):
    GPU = 0
    CPU = -1


class DepthMapProcessor:
    def __init__(self):
        self.depth_pipeline = None

    def process(self, video_path: str) -> (gr.update, str):
        if not video_path:
            raise gr.Error("Wrong video path.")

        if self.depth_pipeline is None:
            if not torch.cuda.is_available():
                logger.error("[Depth Anything V2] CUDA is not available.")
                raise gr.Error("[Depth Anything V2] CUDA is not available.")

            try:
                logger.info("[Depth Anything V2] Loading the model...")
                self.depth_pipeline = pipeline(
                    task="depth-estimation",
                    model=DEPTH_MODEL_NAME,
                    device=DeviceType.GPU
                )
                logger.info("[Depth Anything V2] The model has been uploaded successfully.")
            except Exception as e:
                self.depth_pipeline = None
                logger.error(f"[Depth Anything V2] Error loading the model: {e}")
                raise gr.Error(f"[Depth Anything V2] Error loading the model: {e}")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video_path = os.path.join(OUTPUT_DIR, "output_depth_anything_v2.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # кодек H.264 (AVC). другие: [mp4v, avc1, XVID]
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        logger.info(f"[Depth Anything V2] Video processing has started: {video_path}")

        while cap.isOpened():
            is_successfully_read, frame = cap.read()
            if not is_successfully_read:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            result = self.depth_pipeline(pil_img)
            depth_np = np.array(result["depth"])

            depth_resized = cv2.resize(depth_np, (width, height))

            color_depth = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)  # тепловой градиент
            # color_depth = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)  # чб карта

            out.write(color_depth)

        cap.release()
        out.release()
        logger.info(f"[Depth Anything V2] Processing is completed. The file is saved: {output_video_path}")

        return gr.update(visible=False), output_video_path
