---
applyTo: "**"
---

# Python development and testing

## Coding and review

- Follow idiomatic Python, PEP 8 and the project's existing Ruff configuration.
  Use clear names, type hints for interfaces, context managers for resources,
  and specific exceptions. Add comments/docstrings only where they clarify intent.
- Give functions one coherent responsibility. Extract helpers at meaningful
  boundaries, not to satisfy arbitrary line limits. Prefer simple functions
  over unnecessary classes, wrappers or abstractions.
- Separate pure transformations from file access, subprocesses, network calls
  and model execution so logic can be unit-tested independently.
- Make meaningful user/workflow choices configurable, with existing behavior
  as the default. Name internal constants; do not expose every implementation
  detail as a setting or CLI flag.
- Optimize demonstrated waste, such as repeated decoding or resource loading.
  Do not claim performance gains without measurements.
- For code reviews, inspect local code first. Focus on readability, scope,
  efficiency, useful configuration and test relevance. Do not consult the web
  unless requested or genuinely needed to resolve an external dependency issue.
  Review-only requests do not authorize edits.

## Test layout and execution

- **Default development mode is `rapid`.** Never switch to full mode merely
  because a change could benefit from extra validation.
- **Current restriction: no model execution or long tests.** Treat GPU/model
  execution as unavailable for this development work. Do not try local LLM/TTS/ASR
  inference, model loading/downloads, training, GPU probing/preflights, cloud model
  calls or fallback runtimes. A fake backend is acceptable only with all expensive
  dependencies mocked. Resume such work only after explicit user authorization.
- Prioritize fast iterations. Do not run real media pipelines, full renders,
  integration suites, sleep/polling tests or multi-second workflows to validate
  routine changes. Use targeted units and linting; no exploratory long tests.

- Keep ordinary module `tests` directories unit-only. Put integration tests in
  a dedicated `tests\integration` subfolder, with expensive fixtures there too.
  Model-dependent tests belong there and must also carry the `llm` marker.
- Repository-root `tests` is exclusively external integration: invoke commands
  or use public protocols and inspect outputs/files. Do not import or monkeypatch
  YapIt modules, load source files, or inject source paths there, including fixtures.
  Shared application units belong in `src\tests`; module units stay with their owner.
- Default change validation must be fast unit tests: mock external processes,
  media encoding/decoding, network calls, microphones, GUI loops and model execution.
  Do not add sleeps, polling, full renders or multi-second external workflows
  to the unit suite. Do not recreate the deleted Editor full-render test.
- Do not keep full audio-to-MP4 pipeline-render tests in any suite. When explicitly
  authorized, validate that workflow manually without adding a permanent render test.
- Integration/model tests are explicit opt-in, not routine validation. Do not
  run them after each change, silently enable their flags, or run broad suites
  that pull in unrelated expensive work. Run them only when explicitly requested.
- The repository enforces opt-in with `--run-integration` and `--run-llm`.
  Both also require `--development-mode=full`; they cannot enable expensive
  tests while the default rapid mode remains active.
  Tests in `tests\integration` folders and legacy tests using the expensive
  `enrolled` fixture are automatically classified as integration. Model tests
  in integration folders require both flags. Do not pass either flag without
  explicit user authorization. Finish relocating legacy expensive tests when
  their module is changed; never disguise real workflows as unit tests.
- Test observable functionality and correctness, not source text, exact private
  helper names, temporary filenames, framework behavior or proof of your edits.
  Avoid redundant cases and unnecessarily strict numerical/layout tolerances.
- Run the smallest relevant unit-test selection and existing lint/format checks.
  Repeat only after new changes or failures justify it. State what ran and what
  remained skipped; do not substitute slow integration tests for focused units.

## Agent-maintenance tests

- Put packaging/build smoke checks, README/catalog synchronization, cold-import
  audits, source/file-layout assertions and checks created only to validate
  agent edits in the root `agenttests` folder, not the unit suite.
- `agenttests` is local-only, Git-ignored and untracked. Do not add it to default
  test discovery or stage/commit its files. Remove obsolete/redundant checks;
  wheel/packaging verification is not a product test.
- These are automatically marked `agent` and skipped in rapid mode. Run a
  deliberately selected set only with explicit user authorization outside rapid
  development: `pytest agenttests --development-mode=full`.
- Keep real functionality/correctness tests in the module suites. Reading a
  produced JSON/WAV or checking failure cleanup does not automatically make a
  test an agent check; classify by what the assertion proves.
- Do not add tests of source strings, arbitrary private filenames or fixed helper
  structure to the rapid suite. Relocate existing maintenance checks rather
  than allowing them to constrain normal refactoring.

## Minimal observability

- Keep local runtime/test statistics under Git-ignored `.repo-stats`, separated
  by module. Pytest records phase durations and whole-run time, including
  collection/setup overhead; do not describe only test-call time as total time.
- Replace previous-run statistics at every fresh pytest session or runtime
  process's first measurement. Keep only the current run's measurements,
  including multiple calls/phases; do not append historical runs.
- Child processes belonging to that run must preserve the parent's statistics.
- Store `duration_s` rounded to at most two decimal places.
- Use the shared `@timed` decorator at meaningful operation boundaries instead
  of duplicating timing code or instrumenting trivial/high-frequency helpers.
- Use standard logging to stderr, never stdout where CLI/MCP emits structured
  responses. Record function names, elapsed time, outcomes and exception types,
  not arguments, narration, credentials, model contents or exception messages.
- Metrics must remain lightweight and best-effort: statistics failures must not
  break or alter application behavior. No telemetry server or external service.
