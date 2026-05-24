# main.py
#
# Local voice assistant entry point.

import audio_io as aio
import conversation as convo
import llm
import stt
import tts
import tools


SYSTEM_PROMPT = (
    "You are a concise local voice assistant. "
    "Keep answers short and natural. "
    "All responses must sound natural when read out loud. "
    "Do not use markdown, emojis, emoticons, "
    "asterisks, or special formatting. "
    "Always use tools whenever they are relevant."
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

        reply = None
        
        for _ in range(3):
            generated = llm.generate_reply(
                tokenizer=tokenizer,
                model=model,
                messages=conversation.messages,
                max_new_tokens=1024,
                enable_thinking=False,
                tools=tools.get_tools(),
            )

            tool_call = tools.parse_tool_call(generated)
        
            if tool_call is None:
                reply = generated
                break
        
            print(f"Assistant (tool): {generated}")

            tool_result = tools.run_tool(
                tool_call["name"],
                tool_call.get("arguments"),
            )
        
            conversation.add_assistant_message(generated)
        
            conversation.messages.append(
                {
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": str(tool_result),
                }
            )
            
            print(f"Tool result: {tool_result}")
        
        if reply is None:
            reply = "I could not complete that request."
        
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
