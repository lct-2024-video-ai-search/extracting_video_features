import logging

from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from download_video import download_video

from desc_video import BLIP

from voice_desc import ExtractSpeech

from clear_text import remove_punctuation


# описание приходящих объектов
class Objects(BaseModel):
    vido_url: Union[str, None] = None
    vido_desc: Union[str, None] = None


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



print(blip.process_video(save_path))
print('#########')
print(extract_speech.extract_speech(save_path))



@app.post("/api/get_descriptions")
def get_descriptions(objects: Objects):
    logging.info(f"Началось предсказывание есть ли на фото инфографика")
    save_path = "video_temp.mp4"
    video_url = objects.vido_url
    vido_desc = objects.vido_desc
    _ = download_video(video_url, save_path)
    # получаю описания
    video_movement_desc = blip.process_video(save_path)
    speech_desc = extract_speech.extract_speech(save_path)
    # удаляю лишние символы
    vido_desc = remove_punctuation(vido_desc)
    video_movement_desc = remove_punctuation(video_movement_desc)
    speech_desc = remove_punctuation(speech_desc)

    return JSONResponse(
        content={
            "vido_desc": vido_desc,
            "video_movement_desc": video_movement_desc,
            "speech_desc": speech_desc,
        },
        status_code=200
    )