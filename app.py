import gradio as gr


# функция без эффекта для построения глубин
def depth_map(video_path):
    if video_path is None:
        return None
    return video_path


# функция без эффекта для разделения на слои
def separation_layers(video_path):
    if video_path is None:
        return None
    return video_path

# интерфейс
with gr.Blocks(title="AI Video Processor") as demo:
    gr.Markdown("# Веб-приложение для демонстрации нейросетей по обогащению данных видепотока")
    gr.Markdown("Загрузите ролик и выберите один из доступных методов обработки видео")

    # сетка
    with gr.Row():
        # Левая колонка — Загрузка и управление
        with gr.Column():
            gr.Markdown("Входные данные")
            input_video = gr.Video(label="Перетащите видео сюда")

            # Горизонтальный ряд с кнопками под плеером
            with gr.Row():
                split_btn = gr.Button("Разделение на слои", variant="primary")
                depth_btn = gr.Button("Построение карты глубин", variant="primary")

        # Правая колонка — Проигрывание и скачивание
        with gr.Column():
            gr.Markdown("Результат обработки")
            # Плеер автоматически получит кнопки Play и Download при наличии файла
            output_video = gr.Video(label="Результат")

    # Привязываем триггеры клика к кнопкам
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