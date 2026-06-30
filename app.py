import gradio as gr
import cv2
import numpy as np
from processors.layer_separation.layer_separation import LayerSeparationProcessor
from processors.layer_separation.tap_separation.tap_processor import TAPSeparationProcessor
from processors.depth_map.depth_map import DepthMapProcessor
from processors.layer_separation.bbox_methods.ostrack_processor import OSTrackSeparationProcessor
from argparse import ArgumentParser



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="Server ip address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    return parser.parse_args()


sam_processor = LayerSeparationProcessor()
tap_processor = TAPSeparationProcessor()
depth_processor = DepthMapProcessor()
ostrack_processor = OSTrackSeparationProcessor()

with gr.Blocks(title="AI Video Processor") as demo:
    gr.Markdown("### AI Video Processor")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 1. Upload Video")
            input_video = gr.Video(label="Move video here")

            tracking_method = gr.Radio(
                choices=["SAM2 Video", "TAP (CoTracker + Convex Hull)", "OSTrack (BBox Tracking)"],
                value="SAM2 Video",
                label="Выбор алгоритма трекинга",
                visible=False
            )

            with gr.Row():
                split_btn = gr.Button("Separation into layers", variant="primary")
                depth_btn = gr.Button("Building a depth map", variant="primary")

            first_frame_editor = gr.ImageEditor(
                label="Кликните точно по объекту (красный маркер)",
                type="numpy",
                interactive=True,
                visible=False,
                brush={"colors": ["#FF0000"], "default_size": 1},
                eraser=False,
                sources=[]
            )

            run_track_btn = gr.Button("Запустить отслеживание объекта", variant="stop", visible=False)

        with gr.Column():
            gr.Markdown("#### 2. Processing Result")
            mask_dropdown = gr.Dropdown(
                label="Select Mask Layer",
                choices=[],
                interactive=True,
                visible=False
            )
            output_video = gr.Video(label="Result")


    def prepare_layer_separation(video_path):
        if not video_path:
            raise gr.Error("Сначала загрузите видео!")

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return (
                gr.update(value={"background": frame_rgb, "layers": [], "composite": frame_rgb}, visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        else:
            raise gr.Error("Не удалось прочитать первый кадр видео.")


    split_btn.click(
        fn=prepare_layer_separation,
        inputs=[input_video],
        outputs=[first_frame_editor, run_track_btn, tracking_method]
    )


    def on_run_tracking(video_path, editor_data, method):
        if not editor_data or not editor_data.get("layers"):
            raise gr.Error("Пожалуйста, поставьте хотя бы одну красную точку на объекте!")

        user_layer = editor_data["layers"][0]
        y_indices, x_indices = np.where(user_layer[:, :, 3] > 0)

        if len(x_indices) == 0:
            raise gr.Error("Точки не обнаружены. Кликните по кадру еще раз.")

        all_points = [[int(x), int(y)] for x, y in zip(x_indices, y_indices)]
        sampled_points = all_points[::10] or [all_points[0]]

        print(f"[GRADIO] Используем алгоритм: {method}")

        if method == "SAM2 Video":
            paths = sam_processor.process(video_path, clicked_points=sampled_points)
        elif method == "TAP (CoTracker + Convex Hull)":
            paths = tap_processor.process(video_path, clicked_points=sampled_points)
        elif method == "OSTrack (BBox Tracking)":
            paths = ostrack_processor.process(video_path, clicked_points=sampled_points)
        else:
            paths = []

        if not paths:
            return gr.update(choices=[], visible=False), None

        return gr.update(choices=paths, value=paths[0], visible=True), paths[0]


    run_track_btn.click(
        fn=on_run_tracking,
        inputs=[input_video, first_frame_editor, tracking_method],
        outputs=[mask_dropdown, output_video]
    )


    def run_depth_and_hide_ui(video_path):
        mask_out, video_out = depth_processor.process(video_path)
        return (
            mask_out,
            video_out,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    depth_btn.click(
        fn=run_depth_and_hide_ui,
        inputs=[input_video],
        outputs=[mask_dropdown, output_video, first_frame_editor, run_track_btn, tracking_method]
    )

    mask_dropdown.change(
        fn=lambda x: x,
        inputs=[mask_dropdown],
        outputs=[output_video]
    )

if __name__ == "__main__":
    args = parse_args()
    demo.launch(server_name=args.ip, server_port=args.port)