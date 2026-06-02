# tts.py
#
# Text-to-speech synthesis.

import io
import wave
import numpy as np
import soundfile as sf
import audio_io as aio
from piper import PiperVoice

import json
from onnxruntime import InferenceSession
from misaki import espeak


PIPER_MODEL = "/Piper/en_US-lessac-medium.onnx"

KOKORO_MODEL = "/Kokoro-82M-v1.0-ONNX/onnx/model.onnx"
KOKORO_TOKENIZER = "/Kokoro-82M-v1.0-ONNX/tokenizer.json"
KOKORO_VOICES_DIR = "/Kokoro-82M-v1.0-ONNX/voices/"
SAMPLE_RATE = 24000


class KokoroTTS:
    LANGUAGES = {
        "a": "am_liam.bin",
        "b": "bm_lewis.bin",
        "e": "em_alex.bin",
    }

    def __init__(self, language="a", speed=1.0):
        with open(KOKORO_TOKENIZER, "r", encoding="utf-8") as f:
            tokenizer = json.load(f)

        self.vocab = tokenizer["model"]["vocab"]
        self.speed = np.array([speed], dtype=np.float32)

        self.session = InferenceSession(
            KOKORO_MODEL,
            providers=["CPUExecutionProvider"],
        )

        self.set_language(language)

    def set_language(self, language):
        if language not in self.LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Expected one of: {list(self.LANGUAGES)}"
            )

        self.language = language

        voice_file = (
            f"{KOKORO_VOICES_DIR}/"
            f"{self.LANGUAGES[language]}"
        )

        self.voice = (
            np.fromfile(voice_file, dtype=np.float32)
            .reshape(-1, 1, 256)
        )

        if language == "a":
            self.g2p = espeak.EspeakG2P(language="en-us")
        elif language == "b":
            self.g2p = espeak.EspeakG2P(language="en-gb")
        elif language == "e":
            self.g2p = espeak.EspeakG2P(language="es")

    def synthesize(self, text):
        phonemes, _ = self.g2p(text)

        tokens = [self.vocab[p] for p in phonemes if p in self.vocab]

        if not tokens:
            raise RuntimeError(
                f"No tokens produced. "
                f"text={text!r}, phonemes={phonemes!r}"
            )

        max_tokens = min(509, self.voice.shape[0] - 1)
        tokens = tokens[:max_tokens]

        input_ids = np.array([[0, *tokens, 0]], dtype=np.int64)
        style = self.voice[len(tokens)]

        audio = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "style": style,
                "speed": self.speed,
            },
        )[0]

        audio = np.asarray(audio).squeeze().astype(np.float32)

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = 0.8 * audio / peak

        return audio, SAMPLE_RATE


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
    TESTS = {
        "a": "The quick brown fox jumps over the lazy dog.",
        "b": "The quick brown fox jumps over the lazy dog.",
        "e": "El veloz murciélago hindú comía kiwi."
    }
    device, device_info = aio.choose_device()
    tts = KokoroTTS()
    vad_model = aio.load_silero_vad()

    for lang, text in TESTS.items():
        print(lang, ": ", text)

        tts.set_language(lang)
        audio, sample_rate = tts.synthesize(text)

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

