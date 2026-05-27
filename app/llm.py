# llm.py
#
# Text LLM inference and tool routing.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.environ.get("MODEL_PATH", "/Qwen3.5-9B")


def load_model(model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )

    return tokenizer, model


def generate(
    tokenizer,
    model,
    messages,
    max_new_tokens,
    enable_thinking=False,
    tools=None,
    do_sample=False,
    temperature=None,
    top_p=None,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tools=tools,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def test_llm():
    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise local voice assistant. "
                "Keep responses conversational and brief."
            ),
        },
        {
            "role": "user",
            "content": "Introduce yourself in a short sentence.",
        },
    ]

    reply = generate(
        tokenizer=tokenizer,
        model=model,
        messages=messages,
        max_new_tokens=128,
        enable_thinking=False,
    )

    print(reply)


if __name__ == "__main__":
    test_llm()
