import requests
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from download_video import download_video
import torch
from desc_video import BLIP


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
logging.basicConfig(level=logging.INFO)
logging.info(f"Доступна ли видеокарта? Ответ: {torch.cuda.is_available()}")

blip = BLIP(device=device)



video_url = "https://cdn-st.rutubelist.ru/media/d2/4f/eb01d8d44cb7a462343dd4e5fccf/fhd.mp4"
save_path = "video_temp.mp4"  # Замените на путь и имя файла

download_video(video_url, save_path)


print(blip.process_video(video_path))