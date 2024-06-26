import logging
import hashlib
import datetime

from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from download_video import download_video

from desc_video import BLIP

from voice_desc import ExtractSpeech

from clear_text import remove_punctuation, replace_stop_hashtags

from delete_file import delete_file


# описание приходящих объектов
class Objects(BaseModel):
    video_url: Union[str] = None
    video_desc: Union[str] = None

# описание возвращаемых объектов объектов
class DescriptionRequest(BaseModel):
    video_url: str
    video_desc: str
    video_movement_desc: str
    speech_desc: str

# задание констант
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logging.info(f"Доступна ли видеокарта? Ответ: {torch.cuda.is_available()}")

# извелечь то, что происходит на видео
blip = BLIP(device=device)
# извлечь речь из видео
extract_speech= ExtractSpeech(device=device_1)


@app.post("/api/get_descriptions", response_model=DescriptionRequest)
def get_descriptions(objects: Objects):
    logging.info(f"Началось обработка")

    # Генерируем уникальное имя видео
    current_time = datetime.datetime.utcnow().isoformat()
    hash_object = hashlib.md5(current_time.encode())
    hash_str = hash_object.hexdigest()
    save_path = f"{hash_str}_{current_time}.mp4"

    video_url = objects.video_url
    video_desc = objects.video_desc
    download_correct_flg = download_video(video_url, save_path)
    if not download_correct_flg:
        raise HTTPException(status_code=400, detail="Ошибка загрузки видео")
    
    logging.info(f"Создает описание видео")
    video_movement_desc = blip.process_video(save_path)
    logging.info(f"Создает описание речи")
    speech_desc = extract_speech.extract_speech(save_path)
    # удаляю лишние символы
    video_desc = replace_stop_hashtags(video_desc)
    video_desc = remove_punctuation(video_desc)
    video_movement_desc = remove_punctuation(video_movement_desc)
    speech_desc = remove_punctuation(speech_desc)

    # удалить видео
    _ = delete_file(save_path)

    return JSONResponse(
        content={
            "video_url": video_url,
            "video_desc": video_desc,
            "video_movement_desc": video_movement_desc,
            "speech_desc": speech_desc,
        },
        status_code=200
    )