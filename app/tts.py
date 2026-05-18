# tts.py
#
# Text-to-speech synthesis.

import io
import wave

import numpy as np
import soundfile as sf
import audio_io as aio
from piper import PiperVoice


PIPER_MODEL = "/Piper/en_US-lessac-medium.onnx"


class PiperTTS:
    def __init__(self, model_path=PIPER_MODEL):
        self.voice = PiperVoice.load(model_path)

    def synthesize(self, text):
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        wav_buffer.seek(0)

        audio, sample_rate = sf.read(wav_buffer, dtype="float32")

        if audio.ndim > 1:
            audio = audio[:, 0]

        audio = np.asarray(audio, dtype="float32")

        return audio, sample_rate


def test_tts():
    device, device_info = aio.choose_device()
    tts = PiperTTS()
    vad_model = aio.load_silero_vad()

    audio, sample_rate = tts.synthesize("Hello. This is a voice test.")

    print(f"Generated audio shape: {audio.shape}")
    print(f"Generated sample rate: {sample_rate}")

    audio = aio.resample_audio(
        audio,
        sample_rate,
        int(device_info["default_samplerate"]),
    )

    aio.play_interruptible(
        device,
        device_info,
        vad_model,
        audio,
        int(device_info["default_samplerate"]),
    )


if __name__ == "__main__":

    test_tts()

