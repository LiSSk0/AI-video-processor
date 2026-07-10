import json
import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Annotator")

SCRIPT_DIR = Path(__file__).parent
VIDEO_DIR = SCRIPT_DIR / "videos"

video_files = []
current_index = 0


def load_video_list():
    global video_files
    if not VIDEO_DIR.exists():
        logger.error(f"Папка с видео не найдена: {VIDEO_DIR}")
        return []
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() in video_extensions])
    if not video_files:
        logger.warning("В папке videos нет видеофайлов.")
    return video_files


def get_first_frame(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.error(f"Не удалось прочитать первый кадр: {video_path}")
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Ошибка при чтении видео {video_path}: {e}")
        return None


def get_json_path(video_path):
    return video_path.parent / f"{video_path.stem}_points.json"


def load_points(video_path):
    json_path = get_json_path(video_path)
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data.get("points", [])
        except:
            return []
    return []


def save_points(video_path, points_list):
    json_path = get_json_path(video_path)
    with open(json_path, "w") as f:
        json.dump({"points": points_list}, f, indent=2)
    logger.info(f"Точки сохранены в {json_path}")
    return f"Сохранено {len(points_list)} точек"


def update_display(video_path):
    """Обновляет интерфейс для текущего видео."""
    if video_path is None or not video_path.exists():
        return None, "Видео не найдено", gr.update(visible=False), video_path
    frame = get_first_frame(video_path)
    if frame is None:
        return None, "Не удалось прочитать видео", gr.update(visible=False), video_path
    saved_points = load_points(video_path)
    info = f"Видео: {video_path.name}\nСохранённых точек: {len(saved_points)}"
    return frame, info, gr.update(visible=True), video_path


def on_save_click(editor_data, video_path):
    if editor_data is None:
        gr.Warning("Нет данных редактора.")
        return "Нет данных"
    layers = editor_data.get("layers", [])
    if not layers:
        gr.Warning("Не найдено ни одного слоя с точками.")
        return "Нет точек для сохранения"
    layer = layers[0]
    alpha = layer[:, :, 3]
    y_indices, x_indices = np.where(alpha > 0)
    points_list = [[int(x), int(y)] for x, y in zip(x_indices, y_indices)]
    if not points_list:
        gr.Warning("Не обнаружено ни одной точки. Нажмите на изображение, чтобы поставить точку.")
        return "Нет точек"
    return save_points(video_path, points_list)


def on_next(current_video_state):
    global current_index
    if not video_files:
        gr.Info("Нет видео.")
        return gr.update(), gr.update(), gr.update(), current_video_state
    if current_index + 1 < len(video_files):
        current_index += 1
        video_path = video_files[current_index]
        frame, info, btn_visible, _ = update_display(video_path)
        return frame, info, btn_visible, video_path
    else:
        gr.Info("Это последнее видео.")
        return gr.update(), gr.update(), gr.update(), current_video_state


def on_prev(current_video_state):
    global current_index
    if not video_files:
        gr.Info("Нет видео.")
        return gr.update(), gr.update(), gr.update(), current_video_state
    if current_index - 1 >= 0:
        current_index -= 1
        video_path = video_files[current_index]
        frame, info, btn_visible, _ = update_display(video_path)
        return frame, info, btn_visible, video_path
    else:
        gr.Info("Это первое видео.")
        return gr.update(), gr.update(), gr.update(), current_video_state


def create_interface():
    load_video_list()
    if not video_files:
        raise ValueError("Нет видео для аннотации. Поместите видео в папку 'videos' рядом со скриптом.")

    global current_index
    current_index = 0
    first_video = video_files[current_index]
    first_frame = get_first_frame(first_video)

    with gr.Blocks(title="Аннотация точек для видео") as demo:
        gr.Markdown("## Инструмент для указания точек объекта на первом кадре")
        gr.Markdown("Ставьте **красные точки** на объекте, который нужно отслеживать. Затем нажмите «Сохранить точки».")

        current_video_state = gr.State(first_video)

        with gr.Row():
            with gr.Column(scale=2):
                editor = gr.ImageEditor(
                    label="Первый кадр (кликните для добавления точки)",
                    type="numpy",
                    interactive=True,
                    brush={"colors": ["#FF0000"], "default_size": 3},
                    eraser=False,
                    sources=[],
                    value=first_frame
                )
            with gr.Column(scale=1):
                info_text = gr.Textbox(label="Информация", lines=3, interactive=False)
                save_btn = gr.Button("Сохранить точки", variant="primary")
                save_status = gr.Textbox(label="Статус сохранения", interactive=False)
                with gr.Row():
                    prev_btn = gr.Button("◀ Предыдущее")
                    next_btn = gr.Button("Следующее ▶")

        demo.load(
            fn=update_display,
            inputs=[current_video_state],
            outputs=[editor, info_text, save_btn, current_video_state]
        )

        prev_btn.click(
            fn=on_prev,
            inputs=[current_video_state],
            outputs=[editor, info_text, save_btn, current_video_state]
        )
        next_btn.click(
            fn=on_next,
            inputs=[current_video_state],
            outputs=[editor, info_text, save_btn, current_video_state]
        )

        save_btn.click(
            fn=on_save_click,
            inputs=[editor, current_video_state],
            outputs=[save_status]
        ).then(
            fn=lambda: gr.update(),
            inputs=[],
            outputs=[info_text]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инструмент для аннотации точек на видео.")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP-адрес сервера.")
    parser.add_argument("--port", type=int, default=7861, help="Порт (по умолчанию 7861).")
    args = parser.parse_args()

    demo = create_interface()
    demo.launch(server_name=args.ip, server_port=args.port)