import logging
import torch
import cv2
import numpy as np
import re
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip.models.blip import blip_decoder
from collections import OrderedDict
from transformers import MarianMTModel, MarianTokenizer
from fastapi import HTTPException


logging.basicConfig(level=logging.INFO)


class BLIP:
    def __init__(self, model_path="model_base_capfilt_large.pth", image_size=384, device="cpu"):
        self.device = device
        self.image_size = image_size
        logging.info(f"Началась загрузка BLIP")
        self.model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base').to(self.device)
        self.model.eval()
        logging.info(f"Закончилась загрузка BLIP")
        self.model_translate = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ru').to(self.device)
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ru')

    def generate_description(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            caption = self.model.generate(image, sample=True, num_beams=3, max_length=50, min_length=10)

        inputs = self.tokenizer(caption[0], return_tensors="pt")
        translated_tokens = self.model_translate.generate(**inputs.to(self.device))
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        return translated_text

    def clean_text(self, text):
        # Удаление лишних символов
        text = re.sub(r'[^А-Яа-яA-Za-z\s]', '', text)
        # Удаление строк длиной в 1 символ, не являющихся пробелом
        text = re.sub(r'\b\w\b', '', text)
        # Удаление повторяющихся фраз
        text = re.sub(r'\b(\w+\s+){1,}\b', lambda m: ' '.join(OrderedDict.fromkeys(m.group(0).split())), text)
        return text.lower()

    def process_video(self, video_path, frame_interval=48):
        
        video_capture = cv2.VideoCapture(video_path)

        descriptions = np.array([])
        frame_number = 0
        success = True

        while success:
            try:
                video_capture.read()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Ошибка загрузки видео: {e}")
            
            success, frame = video_capture.read()
            if not success:
                break
            if frame_number % frame_interval == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                description = self.generate_description(image)
                descriptions = np.append(descriptions, description)
            frame_number += 1

        video_capture.release()
        descriptions = ' '.join(descriptions)
        return self.clean_text(descriptions)