# Local Voice Assistant Template

A template to build your own local offline voice assistant with agent capabilities.

This example uses:

* Qwen3.5 (~18 GB VRAM in FP16)
* Parakeet STT
* Kokoro TTS
* Silero VAD

Runs fully locally inside Docker.

---

## Features

* Tool-using agent capabilities
* Voice activity detection
* Speech-to-text
* Conversational memory
* Local LLM inference
* Text-to-speech
* Interruptible playback
* English and Spanish support

---

## Models

Expected local folders:

```text
Qwen3.5-9B/
parakeet-tdt-0.6b-v3-onnx/
Kokoro-82M-v1.0-ONNX/
```

---

## Run

* Clone the models:
```bash
git clone https://huggingface.co/Qwen/Qwen3.5-9B
git clone https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
git clone https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
```

* Build and run
```bash
docker compose build
docker compose run --rm qwen python main.py
```

---

## Notes

* No cloud services are used.
* Models are loaded from local folders.
* Tested with NVIDIA GPU acceleration for the LLM.
* STT currently runs on CPU using ONNX Runtime.
* Models are not included in this repository and retain their respective licenses.
