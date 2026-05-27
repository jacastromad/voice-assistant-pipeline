# conversation.py
#
# Conversation history management.

class Conversation:
    def __init__(self, system_prompt="", max_messages=100):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.messages = []

        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, text):
        self.system_prompt = text

        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = text
        else:
            self.messages.insert(
                0,
                {
                    "role": "system",
                    "content": text,
                },
            )

    def add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})
        self.trim()

    def add_assistant_message(self, text):
        self.messages.append({"role": "assistant", "content": text})
        self.trim()

    def add_tool_message(self, name, text):
        self.messages.append({"role": "tool", "name": name, "content": text})
        self.trim()

    def add_exchange(self, user_text, assistant_text):
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.trim()

    def get_last_user_message(self):
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content", "")
    
        return ""

    def trim(self):
        if len(self.messages) <= self.max_messages:
            return

        if self.messages and self.messages[0]["role"] == "system":
            self.messages = (
                [self.messages[0]]
                + self.messages[-(self.max_messages - 1):]
            )
        else:
            self.messages = self.messages[-self.max_messages:]

    def reset(self):
        self.messages = []

        if self.system_prompt:
            self.set_system_prompt(self.system_prompt)

