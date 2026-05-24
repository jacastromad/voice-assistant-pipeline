# tools.py
#
# Agent tools

from datetime import datetime


def get_datetime():
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current system date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


TOOL_REGISTRY = {
    "get_datetime": get_datetime,
}


def get_tools():
    return TOOLS


def run_tool(name, arguments=None):
    if name not in TOOL_REGISTRY:
        return {
            "error": f"Unknown tool: {name}",
        }

    if arguments is None:
        arguments = {}

    return TOOL_REGISTRY[name](**arguments)


def parse_tool_call(text):
    start = text.find("<tool_call>")
    end = text.find("</tool_call>")

    if start == -1 or end == -1:
        return None

    block = text[start:end]

    function_prefix = "<function="
    function_start = block.find(function_prefix)

    if function_start == -1:
        return None

    function_start += len(function_prefix)
    function_end = block.find(">", function_start)

    if function_end == -1:
        return None

    name = block[function_start:function_end].strip()

    return {
        "name": name,
        "arguments": {},
    }
