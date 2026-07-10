import os
import subprocess
import sys

TARGET_RESOLUTIONS = [
    (1920, 1080),
    (1280, 720),
    (854, 480),
    (640, 360),
    (426, 240),
]

VIDEO_CODEC = "libx264"
CRF = 23
PRESET = "medium"  # balanced (fast, medium, slow, etc.)


def scale_video(input_path, output_path, width, height):
    scale_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease"

    cmd = [
        "ffmpeg",
        "-i", input_path,  # входной файл
        "-vf", scale_filter,  # видеофильтр масштабирования
        "-c:v", VIDEO_CODEC,  # видеокодек
        "-crf", str(CRF),  # качество
        "-preset", PRESET,  # скорость кодирования
        "-c:a", "copy",  # аудио копируем без перекодирования (экономит время)
        "-y",  # перезаписывать выходной файл, если существует
        output_path
    ]

    print(f"Создаю {output_path} ...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"  Готово: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"  Ошибка при обработке {output_path}:")
        print(e.stderr.decode())
        raise


def main():
    if len(sys.argv) < 2:
        print("Использование: python scale_video.py <путь_к_видео>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]  # например, .mp4

    for width, height in TARGET_RESOLUTIONS:
        suffix = f"_{width}x{height}"
        output_name = f"{base_name}{suffix}{ext}"
        output_path = os.path.join(dir_name, output_name)

        if os.path.exists(output_path):
            print(f"Пропускаю (уже существует): {output_path}")
            continue

        scale_video(input_path, output_path, width, height)


if __name__ == "__main__":
    main()