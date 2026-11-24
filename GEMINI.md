# Gemini & Antigravity Workspace Instructions

This workspace uses shared autonomous agent guidelines. All instructions in [AGENTS.md](file:///d:/road_damage/AGENTS.md) apply to Antigravity and Gemini sessions.

## 4-Step Autonomous Coding Workflow
1. **Explore First:** Read relevant files, check directory structure, inspect manifests (`package.json`, `pyproject.toml`, etc.) before making assumptions.
2. **Plan Briefly:** Provide a concise file-by-file plan prior to writing code.
3. **Edit Cleanly:** Apply idiomatic changes following existing project patterns and styles.
4. **Verify Automatically:** Always run project linter, typechecker, and test suites (e.g. `npm test`, `pytest`, `npm run lint`, etc.) after making edits, and resolve any failures until green.

For full rules, environment conventions, and safety guidelines, see [AGENTS.md](file:///d:/road_damage/AGENTS.md).

## Integrated Models & Services
- **Moonshot Kimi 3.1 Pro (via NVIDIA NIM):** Available for multimodal pavement image inspection and deep reasoning via `python -m road_damage.kimi_client "<prompt>" --image "<path>"`.
- **Road Damage Web UI:** Live at `http://127.0.0.1:7860` for interactive testing.
