FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git ffmpeg libsndfile1 \
    portaudio19-dev \
    alsa-utils \
    pulseaudio-utils \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers==5.6.2" \
    "accelerate" \
    "huggingface_hub" \
    "bitsandbytes" \
    "peft" \
    "librosa" \
    "soundfile" \
    "sounddevice" \
    "numpy" \
    "silero-vad" \
    "piper-tts" \
    "onnx-asr[cpu]" \
    "onnxruntime" \
    "misaki[en]"

RUN python -m spacy download en_core_web_sm

CMD ["python", "main.py"]
