import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.tool import ToolCallExecutor
from dais_sdk.types import (
    LlmRequestParams,
    ToolLike,
    BaseMessage, ToolMessage, UserMessage,
)

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")
MAX_TURNS = 12

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When you complete the user task, call attempt_completion()."
)

if not API_KEY:
    raise RuntimeError("API_KEY is required.")


provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)

def read_file(file_path: str) -> str:
    """Read a UTF-8 text file by file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def agent_loop() -> None:
    is_running = True

    def attempt_completion() -> str:
        """Finish the current task and terminate the agent loop."""
        nonlocal is_running
        is_running = False
        return "Task completed."

    llm = LLM("deepseek-v4-flash", provider)
    executor = ToolCallExecutor()

    target_file = os.getenv("TARGET_FILE", "README.md")
    user_prompt = (
        f"请阅读文件 `{target_file}` 的核心内容并进行总结。完成后请调用 attempt_completion 工具。"
    )
    messages: list[BaseMessage] = [UserMessage(content=user_prompt)]
    tools: list[ToolLike] = [read_file, attempt_completion]

    print("[user]", user_prompt)

    for turn in range(1, MAX_TURNS + 1):
        if not is_running:
            break

        print(f"\n=== loop {turn} ===")

        params = LlmRequestParams(
            messages=messages,
            instructions=SYSTEM_PROMPT,
            tools=tools,
            tool_choice="auto",
        )

        assistant = llm.generate_text_sync(params)
        messages.append(assistant)

        if assistant.reasoning_content is not None:
            print("[assistant reasoning]", assistant.reasoning_content)
        print("[assistant]", assistant.content)

        if not assistant.tool_calls:
            print("[log] assistant returned without tool calls.")
            break

        for tool_call in assistant.tool_calls:
            tool = params.find_tool(tool_call.name)
            if tool is None:
                result, error = None, f"Tool not found: {tool_call.name}"
            else:
                result, error, _ = executor.execute_sync(tool, tool_call.arguments)

            tool_message = ToolMessage(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=result,
                error=error,
            )
            messages.append(tool_message)

            status = "error" if error else "result"
            payload = error if error else result
            print(f"[tool:{tool_call.name}] {status}=\n{payload}")
    else:
        print("[stopped] reached MAX_TURNS without completion.")

    print("\nAgent loop terminated.")


if __name__ == "__main__":
    agent_loop()
