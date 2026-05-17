from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import ToolDef

class Toolset(ABC):
    def format_tool_name(self, tool_name: str) -> str:
        sanitized_toolset_name = (
            self.name
            .replace(' ', '_')
            .replace('.', '_')
            .replace(':', '_')
            .replace('/', '_')
        )
        if tool_name.startswith(f"{self.name}__"):
            # already formatted, not to do duplicated formatting here
            return tool_name
        return f"{sanitized_toolset_name}__{tool_name}"

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_tools(self) -> list[ToolDef]: ...
