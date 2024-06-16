import requests
import logging


def download_video(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            # Скачиваем видео по частям
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logging.info(f"Видео успешно скачано и сохранено в {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.info(f"Произошла ошибка при скачивании видео: {e}")
        return False