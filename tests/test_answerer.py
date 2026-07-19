from sparcos_rag.retriever import FusedHit
from sparcos_rag.answerer import answer


class FakeMessages:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kw):
        self.outer.last = kw

        class R:
            content = [type("B", (), {"text": "Risposta [a.md]"})()]

        return R()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages(self)
        self.last = None


def test_empty_hits_short_circuits():
    c = FakeClient()
    a = answer("q", [], c, model="claude-x")
    assert a.text == "Non presente nel cervello."
    assert c.last is None


def test_builds_prompt_and_citations():
    c = FakeClient()
    hits = [FusedHit("a.md", "H", "gatto nero", 0.5)]
    a = answer("che colore?", hits, c, model="claude-x")
    assert a.citations == ["a.md"]
    assert "gatto nero" in c.last["messages"][0]["content"]
    assert c.last["model"] == "claude-x"
