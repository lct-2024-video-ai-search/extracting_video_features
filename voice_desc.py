import moviepy.editor as mp
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import logging
import spacy
from langdetect import detect, LangDetectException
from delete_file import delete_file


logging.basicConfig(level=logging.INFO)

class ExtractSpeech:
    def __init__(self, device):
        self.device = device
        logging.info(f"Началась загрузка desc_audio")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/whisper-small")
        logging.info(f"Закончилась загрузка desc_audio")
        logging.info(f"Началась загрузка словаря")
        self.nlp = spacy.load('ru_core_news_md')
        logging.info(f"Закончилась загрузка словаря")

    
    def is_russian_text(self, text):
        """
        Определяет, является ли текст русским, используя библиотеку langdetect.
        """
        try:
            return detect(text) == 'ru'
        except LangDetectException:
            return False
        
    def is_meaningful_text(self, text):
        """
        Проверяет осмысленность текста, используя модель spaCy ru_core_news_md.
        """
        doc = self.nlp(text)
        # Проверяем, что предложения не пустые и содержат хотя бы одно слово
        meaningful_sentences = [sent for sent in doc.sents if len(sent) > 2 and any(token.is_alpha for token in sent)]
        return len(meaningful_sentences) > 0

    def check_text(self, text):
        """
        Проверяет, является ли текст русским и осмысленным.
        """
        if text.strip() == '':
            logging.debug(f"Text '{text}' is empty or whitespace.")
            return False
        if not self.is_russian_text(text):
            logging.debug(f"Text '{text}' is not detected as Russian.")
            return False
        if not self.is_meaningful_text(text):
            logging.debug(f"Text '{text}' is not meaningful.")
            return False
        all_text = text.split(' ')
        for t in all_text:
            if len(t) > 15:
                return False
        return True
        
        
    def extract_speech(self, video_path):
        # Извлечение аудио с использованием moviepy
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            return ' '
        audio_path = "temp_audio.wav"
        try:
            audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
        except Exception as e:
            logging.info(f"Ошибка при получении информации о файле {video_path}: {e}")
            return ' '

        # Чтение аудио файла
        audio_input, _ = sf.read(audio_path)

        # Ensure audio input has the correct shape
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(axis=1)  # Convert stereo to mono if needed

        # Подготовка входных данных для модели
        input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

        # Прогон через модель
        predicted_ids = self.model.generate(input_features)

        # Расшифровка предсказанных символов в текст
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        meaning_flg = self.check_text(transcription[0])
        _ = delete_file(audio_path)
        if meaning_flg:
            return transcription[0]
        else:
            return ' '