# tools.py
#
# Agent tools.

import re
from datetime import datetime
import sys


def get_datetime():
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M")


def shutdown():
    sys.exit(0)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": (
                "Get the current system date and time. "
                "Use this when the user asks for the current time, current date, "
                "today, tomorrow, yesterday, weekday, or relative date resolution."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown",
            "description": "Shutdown assistant.",
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
    "shutdown": shutdown,
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
    """
    Parse Qwen XML-style tool calls.

    Supported format:

    <tool_call>
    <function=get_datetime>
    </function>
    </tool_call>

    Tool calls with parameters can be added later using:

    <tool_call>
    <function=tool_name>
    <parameter=argument_name>
    argument_value
    </parameter>
    </function>
    </tool_call>
    """

    if not text:
        return None

    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        text,
        flags=re.DOTALL,
    )

    if not tool_match:
        return None

    block = tool_match.group(1)

    function_match = re.search(
        r"<function=([^>\s]+)>",
        block,
        flags=re.DOTALL,
    )

    if not function_match:
        return None

    name = function_match.group(1).strip()

    if name == "none":
        return None

    arguments = {}

    parameter_matches = re.finditer(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
        block,
        flags=re.DOTALL,
    )

    for parameter_match in parameter_matches:
        key = parameter_match.group(1).strip()
        value = parameter_match.group(2).strip()
        arguments[key] = value

    return {
        "name": name,
        "arguments": arguments,
    }


