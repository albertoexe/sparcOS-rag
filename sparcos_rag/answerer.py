from dataclasses import dataclass
from sparcos_rag.retriever import FusedHit

NOT_FOUND = "Non presente nel cervello."


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]


_SYSTEM = (
    "Rispondi SOLO usando il contesto fornito. Cita sempre il file fonte tra parentesi "
    "quadre, es. [nota.md]. Se il contesto non contiene la risposta, scrivi esattamente: "
    f"'{NOT_FOUND}' e nient'altro. Sii breve, in italiano."
)


def _format_context(hits: list[FusedHit]) -> str:
    return "\n\n".join(f"[{h.source_path} # {h.heading}]\n{h.content}" for h in hits)


def answer(query: str, hits: list[FusedHit], client, model: str) -> Answer:
    if not hits:
        return Answer(NOT_FOUND, [])
    prompt = f"Contesto:\n{_format_context(hits)}\n\nDomanda: {query}"
    resp = client.messages.create(
        model=model, max_tokens=1024, system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(getattr(b, "text", "") for b in resp.content)
    citations = list(dict.fromkeys(h.source_path for h in hits))
    return Answer(text, citations)
