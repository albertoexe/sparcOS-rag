# P1.2 — Index Hygiene (denylist + invisible-char cleanup)

Date: 2026-07-23
Status: approved (design), pending implementation plan

## Problem
The loader indexes every `*.md` under the vault. Non-knowledge files (agent config,
templates, already-digested raw sources) get embedded and compete in retrieval,
drowning real answers and hurting effective recall. Invisible characters
(zero-width spaces, BOM) can also pollute chunk text and full-text matching.

## Goal / definition of done
- A conceptual query no longer surfaces config/routing files or `archives/processed/`
  duplicates.
- Re-index + `evaluate` on the 43-question neutral gold set: **hit@10 and MRR unchanged
  or better** (hygiene must not cost recall).
- Denylist is configurable (roadmap requirement), not hardcoded-only.

## Scope (what Alberto chose to exclude)
- **Config/routing** (by filename, any dir): `AGENTS.md`, `CLAUDE.md`, `COMMANDS.md`, `index.md`.
- **System junk** (by dir prefix): `.trash/`, `.obsidian/`, `archives/templates/`.
- **Already-digested sources** (by dir prefix): `archives/processed/`.
- **Kept on purpose:** RAG planning + iteration logs (`playground/Second Brain Setup/00 Planning`,
  `02 Iteration Logs`) — real knowledge, must stay retrievable.

## Design

### Unit 1 — Denylist filter in the loader
- `DEFAULT_DENYLIST` constant with two rule kinds:
  - `dir_prefixes`: `.trash/`, `.obsidian/`, `archives/templates/`, `archives/processed/`
  - `filenames`: `AGENTS.md`, `CLAUDE.md`, `COMMANDS.md`, `index.md`
- `load_vault(root, denylist=None)` — defaults to `DEFAULT_DENYLIST`.
- Helper `_is_denied(rel_path: str, denylist) -> bool` operating on the posix relative path.
- Config-driven override: `Config.denylist` populated from env `INDEX_DENYLIST`
  (comma-separated entries; a trailing `/` means dir-prefix, otherwise filename).
  Empty/unset env → use `DEFAULT_DENYLIST`. CLI `index` passes `cfg.denylist` to `load_vault`.

### Unit 2 — Invisible-character cleanup in the loader
- Extend the existing `.replace("\x00", "")` to also strip zero-width / BOM code points:
  `​ ‌ ‍ ⁠ ﻿`.
- Apply to the raw text before frontmatter parse and before hashing (so the hash reflects
  cleaned content and stale detection stays consistent).

## Interfaces / isolation
- Loader stays a pure function over the filesystem — no DB, no embedder. All Unit-1/Unit-2
  behavior is unit-testable with `tmp_path` fixtures.
- No change to chunker/store/retriever contracts.

## Testing (TDD, no DB)
- `test_loader`:
  - denied dir prefixes excluded; denied filenames excluded in any dir.
  - allowed files (incl. RAG planning notes) kept.
  - `INDEX_DENYLIST` env override respected (adds/replaces entries).
  - zero-width chars and BOM stripped from body and reflected in hash.

## Verification / done-check (external)
1. `pytest -q` green.
2. Re-index the real DB (`sparcos-rag index` — incremental will delete now-denied docs).
3. `sparcos-rag evaluate --k 10` on the 43-Q gold set (`eval/questions_neutral.yaml`):
   compare hit@10 against the recorded baseline (**0.930**). Must be ≥. (Richer metrics —
   hit@1/MRR — live on the `p1.1-reranker` branch's `eval.py`; not required for this done-check.)
4. Spot-check: a conceptual query returns wiki concept pages, not config or `processed/` dupes.

## Constraints
- Tests isolated to `sparcos_rag_test` (never the real DB). Loader tests need no DB.
- No secrets in git. One commit per unit. Do not push to master without asking.

## Risks
- Excluding `index.md` also drops `archives/wiki/index.md` (the catalog). Judged acceptable:
  it is a link list, low knowledge density. Reversible via `INDEX_DENYLIST`.
- If a gold-set expected doc lived only under a denied path, recall would drop. Checked:
  gold set expects `archives/wiki/entities/*`, none under denied paths.
