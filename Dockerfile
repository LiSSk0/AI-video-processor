# базовый образ python 3.10
FROM python:3.10-slim

# ставим ffmpeg для корректного отображения видео в браузере
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем весь код приложения в контейнер
COPY . .

# открываем порт, на котором обычно работает gradio (по умолчанию 7860)
EXPOSE 7860

# задаем переменную окружения, чтобы Gradio был доступен извне контейнера
ENV GRADIO_SERVER_NAME="0.0.0.0"

# команда для запуска приложения
CMD ["python", "app.py"]