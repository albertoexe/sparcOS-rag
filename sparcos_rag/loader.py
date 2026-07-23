from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
import hashlib
import re


@dataclass(frozen=True)
class Document:
    source_path: str
    frontmatter: dict
    body: str
    content_hash: str


_FM = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    fm: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, m.group(2)


DEFAULT_DENYLIST = {
    "dir_prefixes": (".trash/", ".obsidian/", "archives/templates/", "archives/processed/"),
    "filenames": ("AGENTS.md", "CLAUDE.md", "COMMANDS.md", "index.md"),
}


def _is_denied(rel_path: str, denylist: dict) -> bool:
    if any(rel_path.startswith(p) for p in denylist["dir_prefixes"]):
        return True
    return rel_path.rsplit("/", 1)[-1] in denylist["filenames"]


def load_vault(root: Path, denylist: dict | None = None) -> Iterator[Document]:
    if denylist is None:
        denylist = DEFAULT_DENYLIST
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if _is_denied(rel, denylist):
            continue
        raw = path.read_text(encoding="utf-8").replace("\x00", "")
        fm, body = _parse_frontmatter(raw)
        if not body.strip():
            continue
        yield Document(
            source_path=rel,
            frontmatter=fm,
            body=body.strip(),
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
