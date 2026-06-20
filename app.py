import gradio as gr
from processors.layer_separation import separation_layers
from processors.depth_map import depth_map


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
            output_video = gr.Video(label="Result")

    split_btn.click(
        fn=separation_layers,
        inputs=[input_video],
        outputs=[output_video]
    )

    depth_btn.click(
        fn=depth_map,
        inputs=[input_video],
        outputs=[output_video]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)