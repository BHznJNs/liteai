import zipfile
import frontmatter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, cast
from .resource import SkillResource, create_from_bytes as create_resource_from_bytes
from .exceptions import InvalidSkillArchiveError


ZipPath = PurePosixPath

@dataclass
class SkillMd:
    name: str
    description: str
    content: str

    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class SkillParser:
    SKILL_MD_NAME = "skill.md"

    @staticmethod
    def find_skill_root(paths: list[ZipPath]) -> ZipPath | None:
        """
        According to the specification, the skill root should be the first-level directory that contains the SKILL.md file.
        If there is no such directory, the root is the archive root.
        """
        for path in paths:
            parts = path.parts
            if len(parts) == 2 and parts[1].lower() == SkillParser.SKILL_MD_NAME:
                return path.parent
            if len(parts) == 1 and path.name.lower() == SkillParser.SKILL_MD_NAME:
                return path.parent
        return None

    @staticmethod
    def find_skill_md(paths: list[ZipPath], root: ZipPath) -> ZipPath | None:
        for path in paths:
            if path.name.lower() == SkillParser.SKILL_MD_NAME and path.parent == root:
                return path
        return None

    @staticmethod
    def parse_skill_md(text: str) -> SkillMd:
        def resolve_optional_str(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            return str(value)

        result = frontmatter.loads(text)
        return SkillMd(
            name=str(result["name"]),
            description=str(result["description"]),
            content=result.content,

            license=resolve_optional_str(result.get("license")),
            compatibility=resolve_optional_str(result.get("compatibility")),
            allowed_tools=resolve_optional_str(result.get("allowed-tools")),
            metadata=cast(dict[str, Any], result.get("metadata", {})),
        )

@dataclass
class Skill(SkillMd):
    resources: list[SkillResource] = field(default_factory=list)

    @classmethod
    def from_zip(cls, zip_file: zipfile.ZipFile) -> Skill:
        names = zip_file.namelist()
        paths = [ZipPath(name) for name in names]

        skill_root = SkillParser.find_skill_root(paths)
        if skill_root is None:
            raise InvalidSkillArchiveError("Skill root not found")

        skill_md = SkillParser.find_skill_md(paths, skill_root)
        if skill_md is None:
            raise InvalidSkillArchiveError("SKILL.md not found")

        skill_md_text = zip_file.read(str(skill_md)).decode("utf-8-sig", errors="replace")
        skill = SkillParser.parse_skill_md(skill_md_text)

        resources: list[SkillResource] = []
        for info in zip_file.infolist():
            if info.is_dir(): continue
            info_path = ZipPath(info.filename)
            if info_path == skill_md: continue # skip SKILL.md

            relative = info_path.relative_to(skill_root)
            content_bytes = zip_file.read(info.filename)
            resources.append(create_resource_from_bytes(str(relative), content_bytes))

        return cls(
            **skill.__dict__,
            resources=resources,
        )

__all__ = [
    "SkillResource",
    "Skill",
]
