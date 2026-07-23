# Index Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop non-knowledge files and invisible characters from polluting the RAG index, without costing retrieval recall.

**Architecture:** All changes live in the loader (a pure filesystem function) plus config wiring. A `DEFAULT_DENYLIST` filters files by directory prefix and filename; an env var can override it. Invisible/zero-width characters are stripped alongside the existing NUL cleanup. The denylist is threaded identically through `index_vault` and `status_vault` so freshness stays consistent.

**Tech Stack:** Python 3.11, pytest, Postgres/pgvector (not needed for loader tests).

## Global Constraints

- TDD: write the failing test first; loader/config tests use `tmp_path`, never a DB.
- Tests that need a DB run against `sparcos_rag_test` only — never the real DB.
- No secrets in git. One commit per task. Do NOT push to master without asking.
- Follow existing loader style: pure function, posix relative paths, frozen `Document`.

---

### Task 1: Denylist filter in the loader (+ config + wiring)

**Files:**
- Modify: `sparcos_rag/loader.py` (add `DEFAULT_DENYLIST`, `_is_denied`, `denylist` param)
- Modify: `sparcos_rag/config.py` (add `denylist` field + `_parse_denylist`)
- Modify: `sparcos_rag/indexer.py` (thread `denylist` through `index_vault` and `status_vault`)
- Modify: `sparcos_rag/cli.py` (pass `cfg.denylist` in `index` and `status`)
- Test: `tests/test_loader.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `loader.DEFAULT_DENYLIST: dict` = `{"dir_prefixes": tuple[str], "filenames": tuple[str]}`
  - `loader.load_vault(root: Path, denylist: dict | None = None) -> Iterator[Document]`
  - `loader._is_denied(rel_path: str, denylist: dict) -> bool`
  - `config.Config.denylist: dict`
  - `indexer.index_vault(root, embedder, store, batch_size=64, denylist=None)`
  - `indexer.status_vault(root, store, denylist=None)`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_loader.py`)

```python
from pathlib import Path
from sparcos_rag.loader import load_vault, DEFAULT_DENYLIST, _is_denied


def _write(root: Path, rel: str, text: str = "---\nx: 1\n---\nbody here"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_denylist_excludes_dirs_and_filenames(tmp_path):
    _write(tmp_path, "archives/wiki/entities/foo.md")
    _write(tmp_path, "archives/processed/old.md")
    _write(tmp_path, "archives/templates/tmpl.md")
    _write(tmp_path, "playground/proj/AGENTS.md")
    _write(tmp_path, "archives/wiki/index.md")
    _write(tmp_path, ".trash/gone.md")
    paths = {d.source_path for d in load_vault(tmp_path)}
    assert paths == {"archives/wiki/entities/foo.md"}


def test_denylist_keeps_planning_notes(tmp_path):
    rel = "playground/Second Brain Setup/00 Planning/roadmap.md"
    _write(tmp_path, rel)
    assert rel in {d.source_path for d in load_vault(tmp_path)}


def test_denylist_custom_dict_overrides_default(tmp_path):
    _write(tmp_path, "keep/a.md")
    _write(tmp_path, "drop/b.md")
    dl = {"dir_prefixes": ("drop/",), "filenames": ()}
    paths = {d.source_path for d in load_vault(tmp_path, denylist=dl)}
    assert paths == {"keep/a.md"}


def test_is_denied_matches_prefix_and_filename():
    assert _is_denied("archives/processed/x.md", DEFAULT_DENYLIST) is True
    assert _is_denied("a/b/AGENTS.md", DEFAULT_DENYLIST) is True
    assert _is_denied("archives/wiki/concepts/rag.md", DEFAULT_DENYLIST) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_loader.py -q`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_DENYLIST'` / `_is_denied`.

- [ ] **Step 3: Implement the denylist in `sparcos_rag/loader.py`**

Add near the top (after imports), and modify `load_vault`:

```python
DEFAULT_DENYLIST = {
    "dir_prefixes": (".trash/", ".obsidian/", "archives/templates/", "archives/processed/"),
    "filenames": ("AGENTS.md", "CLAUDE.md", "COMMANDS.md", "index.md"),
}


def _is_denied(rel_path: str, denylist: dict) -> bool:
    if any(rel_path.startswith(p) for p in denylist["dir_prefixes"]):
        return True
    return rel_path.rsplit("/", 1)[-1] in denylist["filenames"]
```

Change the `load_vault` signature and loop head:

```python
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
```

- [ ] **Step 4: Add config field + parser in `sparcos_rag/config.py`**

Add `denylist: dict` to the `Config` dataclass (after `top_k: int`). Add the parser and wire it in `load`:

```python
def _parse_denylist(raw: str) -> dict:
    from sparcos_rag.loader import DEFAULT_DENYLIST
    if not raw.strip():
        return DEFAULT_DENYLIST
    dirs, files = [], []
    for entry in raw.split(","):
        e = entry.strip()
        if not e:
            continue
        (dirs if e.endswith("/") else files).append(e)
    return {"dir_prefixes": tuple(dirs), "filenames": tuple(files)}
```

In `load(...)`, add to the returned `Config(...)`:

```python
        denylist=_parse_denylist(e.get("INDEX_DENYLIST", "")),
```

- [ ] **Step 5: Add the config test** (append to `tests/test_config.py`)

```python
def test_denylist_defaults_when_env_absent():
    from sparcos_rag.loader import DEFAULT_DENYLIST
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "ANSWER_MODEL": "claude-x",
    }
    assert load(env).denylist == DEFAULT_DENYLIST


def test_denylist_env_parsed():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "ANSWER_MODEL": "claude-x",
        "INDEX_DENYLIST": "drop/,SKIP.md",
    }
    dl = load(env).denylist
    assert dl["dir_prefixes"] == ("drop/",)
    assert dl["filenames"] == ("SKIP.md",)
```

- [ ] **Step 6: Thread `denylist` through `sparcos_rag/indexer.py`**

Change both signatures and both `load_vault(root)` calls:

```python
def status_vault(root: Path, store, denylist=None) -> StatusReport:
    ...
    for doc in load_vault(root, denylist):
        ...

def index_vault(root: Path, embedder, store, batch_size: int = 64, denylist=None) -> dict:
    ...
    for doc in load_vault(root, denylist):
        ...
```

- [ ] **Step 7: Pass `cfg.denylist` in `sparcos_rag/cli.py`**

In `index`: `stats = index_vault(cfg.vault_path, embedder, store, denylist=cfg.denylist)`
In `status`: `report = status_vault(cfg.vault_path, store, denylist=cfg.denylist)`

- [ ] **Step 8: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (all existing + 6 new tests).

- [ ] **Step 9: Commit**

```bash
git add sparcos_rag/loader.py sparcos_rag/config.py sparcos_rag/indexer.py sparcos_rag/cli.py tests/test_loader.py tests/test_config.py
git commit -m "feat: configurable denylist to keep non-knowledge files out of the index (P1.2)"
```

---

### Task 2: Strip invisible / zero-width characters in the loader

**Files:**
- Modify: `sparcos_rag/loader.py` (add `_clean`, use it in `load_vault`)
- Test: `tests/test_loader.py`

**Interfaces:**
- Produces: `loader._clean(text: str) -> str` (removes NUL, zero-width, BOM)

- [ ] **Step 1: Write the failing test** (append to `tests/test_loader.py`)

```python
def test_strips_zero_width_and_bom(tmp_path):
    body = "he​llo﻿ wor‍ld"
    _write(tmp_path, "a.md", f"---\nx: 1\n---\n{body}")
    doc = next(iter(load_vault(tmp_path)))
    assert doc.body == "hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_loader.py::test_strips_zero_width_and_bom -q`
Expected: FAIL — body still contains the zero-width chars.

- [ ] **Step 3: Implement `_clean` in `sparcos_rag/loader.py`**

Add the constant + helper near `DEFAULT_DENYLIST`:

```python
_INVISIBLE = ("\x00", "​", "‌", "‍", "⁠", "﻿")


def _clean(text: str) -> str:
    for ch in _INVISIBLE:
        text = text.replace(ch, "")
    return text
```

In `load_vault`, replace the read line:

```python
        raw = _clean(path.read_text(encoding="utf-8"))
```

(This subsumes the previous `.replace("\x00", "")`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_loader.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sparcos_rag/loader.py tests/test_loader.py
git commit -m "feat: strip zero-width/BOM characters in the loader (P1.2)"
```

---

### Task 3: Verify on the real index (the external done-check)

**Files:** none (measurement only).

- [ ] **Step 1: Confirm services up**

Run: `docker inspect sparcos-rag-db --format '{{.State.Status}}'` → `running`; `ollama list` responds.

- [ ] **Step 2: Re-index (incremental — deletes now-denied docs)**

Run: `PYTHONUTF8=1 ./.venv/Scripts/sparcos-rag.exe index`
Expected: nonzero `deleted=` count (the config/templates/processed docs), `indexed`/`skipped` for the rest.

- [ ] **Step 3: Freshness clean**

Run: `PYTHONUTF8=1 ./.venv/Scripts/sparcos-rag.exe status`
Expected: `stale=0 nuove=0 rimosse=0` → "pulito". (Confirms status and index filter identically.)

- [ ] **Step 4: Recall did not regress**

Run: `PYTHONUTF8=1 ./.venv/Scripts/sparcos-rag.exe evaluate --k 10` (with `eval/questions_neutral.yaml` — adjust the command's questions file if `evaluate` defaults to `eval/questions.yaml`; run against the 43-Q neutral set).
Expected: `hit@10 >= 0.93`. If lower, STOP — a denied path held a needed doc; revisit the denylist.

- [ ] **Step 5: Spot-check pollution gone**

Run a conceptual query via the `sparcos-rag` MCP `search` tool (or `sparcos-rag query "cos'e un RAG"`).
Expected: results are `archives/wiki/...` knowledge pages; no `AGENTS.md`/`CLAUDE.md`/`index.md`/`archives/processed/*`.

- [ ] **Step 6: Record the result** in the vault iteration log `playground/Second Brain Setup/02 Iteration Logs/2026-07-23.md` (deleted count, hit@10 before/after). No code commit.

---

## Self-Review

**Spec coverage:**
- Denylist (config/routing + junk + processed) → Task 1. ✅
- Configurable via env → Task 1 (`INDEX_DENYLIST`, `_parse_denylist`). ✅
- Invisible-char cleanup → Task 2. ✅
- Keep planning/iteration-log notes → covered by `test_denylist_keeps_planning_notes`. ✅
- Done-check (re-index + evaluate hit@10 ≥ 0.930, spot-check) → Task 3. ✅
- Freshness consistency (status == index filtering) → Task 1 Step 6 + Task 3 Step 3. ✅

**Placeholder scan:** none — all steps carry real code/commands.

**Type consistency:** `denylist` is the same `dict` shape (`dir_prefixes`/`filenames` tuples) across loader, config, indexer. `load_vault(root, denylist)` signature matches all call sites (indexer Task 1 Step 6, tests).

**Open note:** `evaluate` default questions file — the command may point at `eval/questions.yaml`; Task 3 Step 4 flags running against the 43-Q neutral set explicitly.
