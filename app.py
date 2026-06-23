import gradio as gr
import cv2
import numpy as np
from processors.layer_separation.layer_separation import LayerSeparationProcessor
from processors.depth_map.depth_map import depth_map  # Временно старая, пока коллега не обновит
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    # "0.0.0.0", 7860


# Инициализируем процессор слоев один раз при запуске приложения
layer_processor = LayerSeparationProcessor()

with gr.Blocks(title="AI Video Processor") as demo:
    gr.Markdown("### AI Video Processor")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 1. Upload Video")
            input_video = gr.Video(label="Move video here")

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

            run_sam_btn = gr.Button("Запустить отслеживание объекта", variant="stop", visible=False)

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
                gr.update(visible=True)
            )
        else:
            raise gr.Error("Не удалось прочитать первый кадр видео.")

    split_btn.click(
        fn=prepare_layer_separation,
        inputs=[input_video],
        outputs=[first_frame_editor, run_sam_btn]
    )

    def on_run_sam(video_path, editor_data):
        if not editor_data or not editor_data.get("layers"):
            raise gr.Error("Пожалуйста, поставьте хотя бы одну красную точку на объекте!")

        user_layer = editor_data["layers"][0]
        y_indices, x_indices = np.where(user_layer[:, :, 3] > 0)

        if len(x_indices) == 0:
            raise gr.Error("Точки не обнаружены. Кликните по кадру еще раз.")

        all_points = [[int(x), int(y)] for x, y in zip(x_indices, y_indices)]
        sampled_points = all_points[::10] or [all_points[0]]

        print(f"[GRADIO] Передаем в SAM2 массив из {len(sampled_points)} точек.")

        # Читаемый и изолированный вызов декомпозированного класса
        paths = layer_processor.process(video_path, clicked_points=sampled_points)

        if not paths:
            return gr.update(choices=[], visible=False), None
        return gr.update(choices=paths, value=paths[0], visible=True), paths[0]

    run_sam_btn.click(
        fn=on_run_sam,
        inputs=[input_video, first_frame_editor],
        outputs=[mask_dropdown, output_video]
    )

    depth_btn.click(
        fn=depth_map,
        inputs=[input_video],
        outputs=[mask_dropdown, output_video]
    )

    mask_dropdown.change(
        fn=lambda x: x,
        inputs=[mask_dropdown],
        outputs=[output_video]
    )

if __name__ == "__main__":
    args = parse_args()
    demo.launch(server_name=args.ip, server_port=args.port)
