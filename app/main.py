# main.py
#
# Local voice assistant entry point.

import audio_io as aio
import conversation as convo
from llm import load_model, generate
import stt
import tts
from tools import get_tools, parse_tool_call, run_tool


SYSTEM_PROMPT = (
    "You are a concise local voice assistant. "
    "Keep answers short and natural. "
    "All responses must sound natural when read out loud. "
    "Do not use markdown, emojis, emoticons, "
    "asterisks, or special formatting."
)


def build_router_system_prompt(tools):
    tool_blocks = []

    for tool in tools:
        function = tool["function"]
        name = function["name"]
        description = function.get("description", "")

        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})

        parameter_lines = []
        for param_name, param_schema in properties.items():
            param_description = param_schema.get("description", "")
            parameter_lines.append(
                f"<parameter>{param_name}: {param_description}</parameter>"
            )

        parameter_text = "\n".join(parameter_lines)

        tool_blocks.append(
            f"""<tool>
<name>{name}</name>
<description>{description}</description>
{parameter_text}
</tool>"""
        )

    tools_text = "\n\n".join(tool_blocks)

    return f"""You are a tool router.

Your only task is to decide whether the user's request requires a tool.

Do not answer the user.
Do not explain your reasoning.
Do not use JSON.
Do not use Markdown.

If no tool is needed, output exactly:

<tool_call>
<function=none>
</function>
</tool_call>

If a tool is needed, output exactly:

<tool_call>
<function=tool_name>
</function>
</tool_call>

If a tool needs parameters, output exactly:

<tool_call>
<function=tool_name>
<parameter=argument_name>
argument_value
</parameter>
</function>
</tool_call>

Available tools:

<tools>
{tools_text}
</tools>

Rules:
- Output exactly one <tool_call> block.
- Choose exactly one function.
- Use <function=none> only if no tool is required.
- Never answer the user directly.
- Never include text before or after the tool call.
"""


def build_router_messages(conversation, tools):
    return [
        {
            "role": "system",
            "content": build_router_system_prompt(tools),
        },
        {
            "role": "user",
            "content": conversation.get_last_user_message(),
        },
    ]


def main():
    device, device_info = aio.choose_device()

    vad_model = aio.load_silero_vad()
    tokenizer, model = load_model()
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

        if not user_text.strip():
            continue

        conversation.add_user_message(user_text)

        print(f"\nUser: {user_text}")

        router_messages = build_router_messages(conversation, get_tools())

        router_output = generate(
            tokenizer=tokenizer,
            model=model,
            messages=router_messages,
            max_new_tokens=128,
            enable_thinking=False,
            tools=None,
            do_sample=False,
        )

        tool_call = parse_tool_call(router_output)

        if tool_call:
            tool_result = run_tool(
                tool_call["name"],
                tool_call.get("arguments", {}),
            )

            conversation.add_assistant_message(router_output)
            print(f"Tool call: {router_output}")
            conversation.add_tool_message(tool_call["name"], tool_result)
            print(f"Tool: {tool_result}")


        reply = generate(
            tokenizer=tokenizer,
            model=model,
            messages=conversation.messages,
            max_new_tokens=512,
            enable_thinking=False,
            tools=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        conversation.add_assistant_message(reply)

        print(f"Assistant: {reply}")

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

