FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y -q --no-install-recommends libgomp1 ffmpeg libsm6 libxext6 pdftohtml git ninja-build g++

WORKDIR /app

COPY services/doctr/download_models.py .
COPY services/doctr/main.py .
COPY services/doctr/requirements.txt .

RUN pip install -r requirements.txt
RUN python3 download_models.py

EXPOSE 8000

ENTRYPOINT ["python3", "main.py"]
