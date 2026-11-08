---
applyTo: "**"
---

# Dependency management

- Reuse the existing environment. Install dependencies only after a dependency
  file changes or validation fails because a package is missing.
- On `DESKTOP-A05R1KA`, VPN restrictions can block PyPI downloads. Use the approved
  feed (`https://packagefeedproxy.microsoft.io/pypi/simple/`) or verified local
  wheels; keep feed settings local to the command or environment.
- For a missing wheel, use `scripts\cache-feed-wheel.py` and verify its published
  SHA256. Offline wheels must match the target Python, Windows architecture and
  CPU/CUDA runtime, including required dependencies.
- Never bypass TLS checks, change VPN policy, use unapproved mirrors or save
  credentials in tracked files. Report blocked downloads instead.
- Keep CosyVoice in its separate Python 3.10 environment; do not change the
  working Qwen `.venv-cpu` or `.venv-gpu` to resolve its dependencies.
- For agent-selected models, verify the latest appropriate stable release from
  official sources and require a 2026-or-later model release. Prefer dated model
  identifiers; a recent repository edit is not a model release. Do not download,
  replace checkpoints or retrain without authorization.
