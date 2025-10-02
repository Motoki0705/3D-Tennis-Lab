# Vibecoding Standard Workflow (Python)

## Overview

```
Plan → Implement → Test Design/Update → Run Tests → Fix Plan → Implement → … (Test Loop) … → Clear → Report
```

- **Always plan before each fix** (no coding without a hypothesis).
- **Loop continues until all tests pass**.
- **Keep changes minimal** and focused.
- **Documentation and environment references are mandatory** (see below).

---

## References & Rules

- **Execution Environment**: See `docs/uv/how_to_use.md`.
- **Development Guidelines**: Follow `docs/development/development_architecture_guide_ja_rev.md`.
- **Datasets**: Refer to documentation under `docs/dataset/`.
- **Agent Artifacts**:

  - Store `plan.md`, `fix_plan_*.md`, `report.md`, etc. inside `docs/agents/<project_name>/`.
  - Example:

    ```
    docs/agents/ball_tracking_vit_heatmap/
      plan.md
      fix_plan_v1.md
      fix_plan_v2.md
      ...
      report.md
    ```

  - Organize by item folders and version each file to make development history traceable.

---

## Steps

### 1. Plan

Define the objective, scope, and validation before coding.

Template:

```
# PLAN
## Objective
- What is the goal?

## Scope
- In-scope:
- Out-of-scope:

## Validation
- Which tests to add or update
```

📌 Save this as `docs/agents/<project_name>/plan.md`.

---

### 2. Implement

- Apply the smallest change that matches the Plan.
- Avoid unrelated refactoring.

---

### 3. Test

- Add or update unit tests first.
- Use **pytest** (`pytest -q`).
- Run in pyramid order: unit → integration → end-to-end.

---

### 4. Run Tests

- Record failures, logs, and stack traces.
- **Do not fix immediately** → go back to Fix Plan.

---

### 5. Fix Plan

Always analyze before modifying code.

Template:

```
# FIX PLAN
## Failure
- Failed test names and errors

## Hypothesis
- Likely cause(s)

## Change Design
- Files/functions to modify
- Smallest viable fix

## Validation
- Specific tests to re-run
```

📌 Save as versioned files:
`docs/agents/<project_name>/fix_plan_v1.md`, `fix_plan_v2.md`, etc.

---

### 6. Test Loop

- Cycle: Implement → Test → Fix Plan until green.
- If hypothesis fails twice → return to full Plan step.

---

### 7. Clear

- Confirm all tests are green.
- Remove dead code, TODOs, and apply lint/type checks.

---

### 8. Report

Summarize the outcome.

Template:

```
# REPORT
## What changed
- Summary of modifications

## Why
- Reason for changes

## Evidence
- Test results (all pass)

## Follow-ups
- Remaining risks or tasks
```

📌 Save as `docs/agents/<project_name>/report.md`.

---

## Helper Script (Python / Bash)

`scripts/test_loop.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[1] Unit subset"
pytest -q tests/unit || { echo "❌ Unit failed. Update FIX_PLAN."; exit 1; }

echo "[2] Integration"
pytest -q tests/integration || { echo "❌ Integration failed. Update FIX_PLAN."; exit 1; }

echo "[3] End-to-end"
pytest -q tests/e2e || { echo "❌ E2E failed. Update FIX_PLAN."; exit 1; }

echo "✅ All green!"
```

---

## Key Principles

- **Plan before every fix** → no blind coding.
- **Test-driven loop** → only move forward when tests pass.
- **Minimal changes** → keep iterations fast and focused.
- **Version all artifacts** → maintain development history inside `docs/agents/`.
