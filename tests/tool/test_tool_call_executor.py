import pytest

from dais_sdk.tool.tool_call_executor import ToolCallExecutor
from dais_sdk.types import TextBlock, ToolExecutionError


class StubContentBlockPersister:
    def __init__(self):
        self.blocks = []

    async def persist(self, content_block):
        self.blocks.append(content_block)
        return {"id": f"block_{len(self.blocks)}"}


@pytest.mark.asyncio
async def test_execute_string_result_without_content_block_persister():
    def say() -> str:
        return "ok"

    outcome = await ToolCallExecutor().execute(say, {})

    assert outcome.result == "ok"
    assert outcome.error is None
    assert outcome.raw_result == "ok"


@pytest.mark.asyncio
async def test_execute_empty_content_block_list_without_content_block_persister():
    def generate_empty() -> list:
        return []

    outcome = await ToolCallExecutor().execute(generate_empty, {})

    assert outcome.result == []
    assert outcome.error is None
    assert outcome.raw_result == []


@pytest.mark.asyncio
async def test_execute_persists_content_blocks_to_metadata():
    content_block = TextBlock(text="hello")

    def generate_content_blocks():
        return [content_block]

    persister = StubContentBlockPersister()
    outcome = await ToolCallExecutor(content_block_persister=persister).execute(generate_content_blocks, {})

    assert outcome.result == [{"id": "block_1"}]
    assert outcome.error is None
    assert outcome.raw_result == [content_block]
    assert persister.blocks == [content_block]


@pytest.mark.asyncio
async def test_execute_preserves_content_block_metadata_order():
    content_blocks = [TextBlock(text="a"), TextBlock(text="b")]

    def generate_content_blocks():
        return content_blocks

    outcome = await ToolCallExecutor(content_block_persister=StubContentBlockPersister()).execute(generate_content_blocks, {})

    assert outcome.result == [{"id": "block_1"}, {"id": "block_2"}]
    assert outcome.error is None
    assert outcome.raw_result == content_blocks


@pytest.mark.asyncio
async def test_execute_non_empty_content_blocks_without_persister_returns_error():
    def generate_content_blocks():
        return [TextBlock(text="hello")]

    executor = ToolCallExecutor()
    executor.exception_handler.set_handler(ToolExecutionError, lambda e: str(e.raw_error))

    outcome = await executor.execute(generate_content_blocks, {})

    assert outcome.result is None
    assert outcome.error is not None
    assert outcome.raw_result is None
