# Security Policy

## Reporting a vulnerability

Report security issues privately rather than opening a public issue. Include a
description, reproduction steps, and any proposed fix. Do not include real
customer data or credentials in any report.

## Security posture

- **Governed tools, fail-closed.** Every tool enforces the caller's scope
  (admin / manager / region / anonymous) and returns zero rows for unknown
  claims. A misconfigured caller sees nothing, never everything.
- **Human-in-the-loop by default.** The graph interrupts before ANY tool
  execution and resumes only on explicit approval. A rejected call produces a
  rejection message, not a workaround.
- **Masked sensitive fields.** Account numbers are masked (`****-0001`) in
  every tool output, including admin calls.
- **Honest disclosure.** Tool results carry `visible_count` /
  `withheld_count` / `total_count`, so the agent can distinguish "no data"
  from "data you are not allowed to see" — the structural defense against
  fabrication.
- **Synthetic data only.** The governed dataset is generated from constants;
  no real customer data is involved.
- **No secrets in the repo or image.** API keys are read from the environment
  at runtime (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). See `.env.example`.
- **Checkpoints contain conversation data.** The SQLite checkpointer persists
  full message history per thread by design (that is what makes resume
  possible). In production: treat the checkpoint database as sensitive, put it
  on an encrypted volume, and add a retention policy — noted in
  `docs/architecture.md`.

## Supported versions

Only the latest `main` branch is supported for security fixes.
