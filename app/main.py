# main.py
#
# Local voice assistant entry point.

import audio_io as aio
import conversation as convo
import llm
import stt
import tts


SYSTEM_PROMPT = (
    "You are a concise local voice assistant. "
    "Keep responses conversational and brief. "
    "Respond using plain spoken text only. "
    "Do not use markdown, emojis, emoticons, "
    "asterisks, or special formatting."
)


def main():
    device, device_info = aio.choose_device()

    vad_model = aio.load_silero_vad()
    tokenizer, model = llm.load_model()
    tts_engine = tts.PiperTTS()

    conversation = convo.Conversation(
        system_prompt=SYSTEM_PROMPT,
        max_messages=50,
    )

    sample_rate = int(device_info["default_samplerate"])

    print("Ready.")

    while True:
        utterance = aio.capture_utterance(
            device,
            device_info,
            vad_model,
        )

        user_text = stt.transcribe(
            utterance,
            sample_rate,
        )

        print(f"\nUser: {user_text}")

        conversation.add_user_message(user_text)

        reply = llm.generate_reply(
            tokenizer=tokenizer,
            model=model,
            messages=conversation.messages,
            max_new_tokens=1024,
            enable_thinking=False
        )

        print(f"Assistant: {reply}")

        conversation.add_assistant_message(reply)

        audio, audio_sample_rate = tts_engine.synthesize(reply)

        playback_sample_rate = sample_rate

        if audio_sample_rate != playback_sample_rate:
            audio = aio.resample_audio(
                audio,
                audio_sample_rate,
                playback_sample_rate,
            )

        aio.play_interruptible(
            device,
            device_info,
            vad_model,
            audio,
            playback_sample_rate,
        )


if __name__ == "__main__":
    main()
