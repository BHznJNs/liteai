import asyncio
import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.types import LlmRequestParams, UserMessage

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

async def start_task(id: int, prompts: list[str]):
    provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
    llm = LLM("deepseek-v3.1", provider)

    messages = []

    for prompt in prompts:
        messages.append(UserMessage(content=prompt))
        response = await llm.generate_text(
            LlmRequestParams(messages=messages)
        )
        messages.append(response)
        print(f"[task {id} | assistant]", response.content)
        if response.reasoning_content:
            print(f"[task {id} | reasoning]", response.reasoning_content)
        if response.usage:
            print(
                f"[task {id} | usage]",
                f"input={response.usage.input_tokens}",
                f"output={response.usage.output_tokens}",
                f"total={response.usage.total_tokens}",
            )
        await asyncio.sleep(3)

async def main():
    await asyncio.gather(
        start_task(1, ["请用一句话介绍 Python 编程语言", "Python 相对于其它语言有哪些特点"]),
        start_task(2, ["请用一句话介绍 JavaScript 编程语言", "JavaScript 相对于其它语言有哪些特点"]),
        start_task(3, ["请用一句话介绍 Rust 编程语言", "Rust 相对于其它语言有哪些特点"]),
    )

asyncio.run(main())
