# stt.py
#
# Speech-to-text transcription.

import numpy as np
import onnx_asr

import audio_io as aio


PARAKEET_MODEL = "/parakeet-tdt-0.6b-v3-onnx"

model = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    PARAKEET_MODEL,
)


def transcribe(audio, sample_rate):
    audio_16k = aio.resample_to_vad_rate(audio, sample_rate)
    audio_16k = np.asarray(audio_16k, dtype="float32")

    result = model.recognize(
        audio_16k,
        sample_rate=aio.VAD_SR,
    )

    return str(result).strip()


def test_stt():
    device, device_info = aio.choose_device()
    vad_model = aio.load_silero_vad()

    utterance = aio.capture_utterance(
        device,
        device_info,
        vad_model,
    )

    sample_rate = int(device_info["default_samplerate"])

    text = transcribe(utterance, sample_rate)

    print("\nTranscription:")
    print(text)


if __name__ == "__main__":
    test_stt()
