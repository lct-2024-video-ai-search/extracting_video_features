FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 curl gnupg
RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY . /app
RUN pip install transformers timm fairscale
RUN pip install "uvicorn[standard]" fastapi
RUN wget -P /app https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth
RUN pip install https://huggingface.co/spacy/ru_core_news_md/resolve/main/ru_core_news_md-any-py3-none-any.whl
RUN pip install opencv-python moviepy soundfile spacy langdetect sentencepiece

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]