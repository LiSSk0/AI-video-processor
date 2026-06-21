import gradio as gr
from processors.layer_separation.layer_separation import separation_layers
from processors.depth_map.depth_map import depth_map


with gr.Blocks(title="AI Video Processor") as demo:
    gr.Markdown("A web application for demonstrating neural networks for enriching video stream data")
    gr.Markdown("Upload your video and select one of the available video processing methods.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("Input data")
            input_video = gr.Video(label="Move video here")

            with gr.Row():
                split_btn = gr.Button("Separation into layers", variant="primary")
                depth_btn = gr.Button("Building a depth map", variant="primary")

        with gr.Column():
            gr.Markdown("Processing result")

            mask_dropdown = gr.Dropdown(
                label="Select Mask Layer",
                choices=[],
                interactive=True,
                visible=False
            )

            output_video = gr.Video(label="Result")


    # посредник для обработки завершения разделения слоев
    def on_split_complete(video_p):
        paths = separation_layers(video_p)
        if not paths:
            return gr.update(choices=[], visible=False), None
        return gr.update(choices=paths, value=paths[0], visible=True), paths[0]


    split_btn.click(
        fn=on_split_complete,
        inputs=[input_video],
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
    demo.launch(server_name="0.0.0.0", server_port=7860)