# Shared Autonomous Coding Rules (AGENTS.md)

This file defines shared guidelines and execution standards for **DeepSeek Harness (DSH)**, **Antigravity**, and any AI coding agents operating within this workspace.

---

## 1. Core Operating Principles

Agents must follow a strict **Claude Code-style 4-phase loop**:
```
Explore ──> Plan ──> Edit ──> Verify
```

### Phase 1: Explore First
- **Investigate before acting:** Read existing files, directory structures, and configurations before making changes or proposing code.
- **Search first:** Use search/grep tools to locate definitions, usages, and existing design patterns.
- **Inspect dependencies:** Check dependency manifests (`package.json`, `pyproject.toml`, `requirements.txt`) to know available libraries.

### Phase 2: Plan Briefly
- **Output a concise file-by-file plan:** Before modifying or creating code, outline the target files and the specific modifications planned.
- **Scope discipline:** Keep the scope strictly bounded to the user request. Avoid unnecessary rewrites or scope creep.

### Phase 3: Edit Cleanly
- **Idiomatic diffs:** Write clean, minimal code matching existing repository conventions and idioms.
- **Preserve style:** Respect project formatting, naming conventions, and linting standards.
- **No placeholder code:** Write complete implementations; do not leave `TODO`, `...`, or stubbed out logic without explicit request.

### Phase 4: Verify Automatically
- **Run validation tools:** Always run linters, typecheckers, and tests after making edits.
- **Auto-fix failures:** If any test, typecheck, or lint error occurs, inspect the output, pinpoint the root cause, and iterate until the verification commands pass cleanly.
- **Do not declare completion on red:** A task is not done until verification passes.

---

## 2. Environment & Project Stack

- **Node.js Runtime:** Node.js v22+ (`node -v`, `npm -v`, `npx`)
- **Python Runtime:** Python 3.12+ (`python -v`)
- **Version Control:** Git

### Recommended Verification Commands

#### For JavaScript / TypeScript (DSH & Node tooling):
- **Test:** `npm test`
- **Lint:** `npm run lint` or `npx eslint .`
- **Build / CLI Verification:** `npx @deepseek-ai/dsh --help`

#### For Python:
- **Test:** `pytest` or `python -m unittest` (or `.\.venv\Scripts\python -m pytest` when virtualenv is configured)
- **Lint:** `ruff check .` or `flake8`
- **Typecheck:** `mypy .`

---

## 3. Safety & Credential Hygiene
- **Never expose secrets:** Never print, log, or commit API keys (such as `DEEPSEEK_API_KEY`).
- **Environment variables:** Store secrets in `.env` (kept in `.gitignore`). Reference templates in `.env.example`.
- **Destructive operations:** Never delete datasets, databases, or entire directories without explicit user confirmation.
