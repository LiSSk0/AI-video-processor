import os
import subprocess
import sys
import glob

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


def scale_video(input_path, output_path, max_w, max_h):
    # a = соотношение сторон (width/height)
    scale_filter = (
        f"scale='if(gt(a,{max_w}/{max_h}),{max_w},trunc({max_h}*a/2)*2):"
        f"if(gt(a,{max_w}/{max_h}),trunc({max_w}/a/2)*2,{max_h})'"
    )

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", scale_filter,
        "-c:v", VIDEO_CODEC,
        "-crf", str(CRF),
        "-preset", PRESET,
        "-c:a", "copy",
        "-y",
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(script_dir, "Birds_02___4K_res.*")
    files = glob.glob(pattern)

    if not files:
        print(f"Файл Birds_02___4K_res не найден в {script_dir}")
        sys.exit(1)

    input_path = files[0]
    print(f"Обрабатываю файл: {input_path}")

    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]

    for max_w, max_h in TARGET_RESOLUTIONS:
        suffix = f"_{max_w}x{max_h}"
        output_name = f"{base_name}{suffix}{ext}"
        output_path = os.path.join(dir_name, output_name)

        if os.path.exists(output_path):
            print(f"Пропускаю (уже существует): {output_path}")
            continue

        scale_video(input_path, output_path, max_w, max_h)


if __name__ == "__main__":
    main()