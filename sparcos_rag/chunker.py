from dataclasses import dataclass
import re
from sparcos_rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    source_path: str
    heading: str
    chunk_index: int
    content: str


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(body: str):
    heading, buf = "", []
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            if buf:
                yield heading, "\n".join(buf).strip()
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        yield heading, "\n".join(buf).strip()


def _split_len(text: str, max_chars: int):
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_document(doc: Document, max_chars: int = 1500) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for heading, content in _sections(doc.body):
        if not content:
            continue
        for piece in _split_len(content, max_chars):
            chunks.append(Chunk(doc.source_path, heading, idx, piece))
            idx += 1
    return chunks
