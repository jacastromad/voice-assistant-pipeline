# audio_io.py
#
# Microphone input, speaker playback, and VAD utilities.

import time

import librosa
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

VAD_SR = 16000
VAD_CHUNK_SAMPLES = 512
SPEECH_THRESHOLD = 0.5
SILENCE_SECONDS = 0.8


def choose_device(device_id=None):
    devices = sd.query_devices()

    if device_id is not None:
        if not isinstance(device_id, int):
            raise TypeError("device_id must be an integer")

        if device_id < 0 or device_id >= len(devices):
            raise ValueError(f"Invalid device id: {device_id}")

        if devices[device_id]["max_input_channels"] <= 0:
            raise ValueError(f"Device {device_id} is not an input device")

        return device_id, devices[device_id]

    valid_devices = []

    print("\nAvailable input devices:\n")

    for idx, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            valid_devices.append(idx)

            print(f"[{idx}] {device['name']}")
            print(f"     Input channels : {device['max_input_channels']}")
            print(f"     Default SR     : {int(device['default_samplerate'])}")
            print()

    if not valid_devices:
        raise RuntimeError("No input devices found.")

    while True:
        try:
            device_id = int(input("Select input device id: ").strip())

            if device_id not in valid_devices:
                print("Invalid input device id.\n")
                continue

            return device_id, devices[device_id]

        except ValueError:
            print("Please enter a numeric device id.\n")


def resample_audio(audio, source_sample_rate, target_sample_rate):
    if source_sample_rate == target_sample_rate:
        return audio.copy()

    source_len = len(audio)
    target_len = int(source_len * target_sample_rate / source_sample_rate)

    source_positions = torch.linspace(
        0,
        source_len - 1,
        target_len,
    )

    left = torch.floor(source_positions).long()
    right = torch.clamp(left + 1, max=source_len - 1)

    weight = source_positions - left

    audio_tensor = torch.from_numpy(audio).float()

    resampled = (
        audio_tensor[left] * (1.0 - weight)
        + audio_tensor[right] * weight
    )

    return resampled.numpy().astype("float32")


def resample_to_vad_rate(audio, sample_rate):
    if sample_rate == VAD_SR:
        return audio.copy()

    if sample_rate == 48000:
        return audio[::3].copy()

    return librosa.resample(audio, orig_sr=sample_rate, target_sr=VAD_SR)


def fix_vad_chunk_size(audio_16k):
    original_len = len(audio_16k)

    if len(audio_16k) > VAD_CHUNK_SAMPLES:
        audio_16k = audio_16k[:VAD_CHUNK_SAMPLES]

    tensor = torch.from_numpy(audio_16k).float()

    if tensor.numel() < VAD_CHUNK_SAMPLES:
        tensor = torch.nn.functional.pad(
            tensor,
            (0, VAD_CHUNK_SAMPLES - tensor.numel())
        )

    return tensor, original_len


def chunk_samples_for_rate(sample_rate):
    return int(sample_rate * (VAD_CHUNK_SAMPLES / VAD_SR))


def vad_probability(vad_model, audio, sample_rate):
    audio_16k = resample_to_vad_rate(audio, sample_rate)
    tensor, _ = fix_vad_chunk_size(audio_16k)
    return vad_model(tensor, VAD_SR).item()


def capture_utterance(device, device_info, vad_model):
    sample_rate = int(device_info["default_samplerate"])
    chunk_samples = chunk_samples_for_rate(sample_rate)

    speaking = False
    last_speech_time = None
    buffer = []

    print("\nListening...")

    with sd.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
    ) as stream:
        while True:
            audio, _ = stream.read(chunk_samples)
            audio = audio[:, 0]

            prob = vad_probability(vad_model, audio, sample_rate)
            now = time.time()

            if prob > SPEECH_THRESHOLD:
                last_speech_time = now

                if not speaking:
                    speaking = True
                    buffer = []
                    print("Speech started")

                buffer.append(audio.copy())

            elif speaking:
                buffer.append(audio.copy())

                if last_speech_time and now - last_speech_time > SILENCE_SECONDS:
                    print("Speech stopped")

                    if not buffer:
                        return audio

                    return torch.cat([
                        torch.from_numpy(chunk)
                        for chunk in buffer
                    ]).numpy()


def play_interruptible(device, device_info, vad_model, audio, playback_sample_rate):
    input_sample_rate = int(device_info["default_samplerate"])
    input_chunk_samples = chunk_samples_for_rate(input_sample_rate)
    output_block_samples = chunk_samples_for_rate(playback_sample_rate)

    with sd.InputStream(
        device=device,
        samplerate=input_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=input_chunk_samples,
    ) as mic_stream, sd.OutputStream(
        device=device,
        samplerate=playback_sample_rate,
        channels=audio.shape[1] if audio.ndim > 1 else 1,
        dtype="float32",
        blocksize=output_block_samples,
    ) as out_stream:
        pos = 0

        while pos < len(audio):
            mic_audio, _ = mic_stream.read(input_chunk_samples)
            mic_audio = mic_audio[:, 0]

            prob = vad_probability(vad_model, mic_audio, input_sample_rate)
            peak = float(abs(mic_audio).max())

            if prob > SPEECH_THRESHOLD and peak > 0.02:
                return True

            block = audio[pos:pos + output_block_samples]

            if len(block) < output_block_samples:
                padding_shape = (
                    output_block_samples - len(block),
                    audio.shape[1] if audio.ndim > 1 else 1,
                )

                padding = torch.zeros(padding_shape).numpy()

                if audio.ndim == 1:
                    block = torch.cat([
                        torch.from_numpy(block),
                        torch.from_numpy(padding[:, 0]),
                    ]).numpy()
                else:
                    block = torch.cat([
                        torch.from_numpy(block),
                        torch.from_numpy(padding),
                    ]).numpy()

            out_stream.write(block)
            pos += output_block_samples

    return False


def test_vad(device, device_info):
    sample_rate = int(device_info["default_samplerate"])
    chunk_samples = int(sample_rate * (VAD_CHUNK_SAMPLES / VAD_SR))

    model = load_silero_vad()


    speaking = False
    last_speech_time = None

    print("\nListening. Speak, then stop. Ctrl+C to exit.\n")

    try:
        with sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            while True:
                audio, _ = stream.read(chunk_samples)
                audio = audio[:, 0]

                audio_16k = resample_to_vad_rate(audio, sample_rate)
                tensor, resampled_len_before_fix = fix_vad_chunk_size(audio_16k)

                prob = model(tensor, VAD_SR).item()
                now = time.time()

                is_speech = prob > SPEECH_THRESHOLD

                if is_speech:
                    last_speech_time = now

                    if not speaking:
                        speaking = True
                        print(f"Speech started (prob={prob:.3f})")

                elif speaking and last_speech_time:
                    if now - last_speech_time > SILENCE_SECONDS:
                        speaking = False
                        print(f"Speech stopped  (prob={prob:.3f})")

    except KeyboardInterrupt:
        print("Stopped.")


def test_audio_io(device, device_info):
    vad_model = load_silero_vad()

    utterance = capture_utterance(
        device,
        device_info,
        vad_model,
    )

    duration = (
        len(utterance)
        / int(device_info["default_samplerate"])
    )

    print(f"Captured {duration:.2f} seconds")

    audio = torch.stack([
        torch.from_numpy(utterance),
        torch.from_numpy(utterance),
    ], dim=1).numpy()

    interrupted = play_interruptible(
        device,
        device_info,
        vad_model,
        audio,
        int(device_info["default_samplerate"]),
    )

    if interrupted:
        print("Playback interrupted.")
    else:
        print("Playback completed.")


if __name__ == "__main__":
    device, device_info = choose_device()
    test_vad(device, device_info)
    test_audio_io(device, device_info)

