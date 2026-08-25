# Domain Docs

This is a single-context repository.

## Before exploring

- Read `CONTEXT.md` at the repository root.
- Read the ADRs under `docs/adr/` that touch the area being changed.
- If a document does not exist, proceed silently; domain documentation is created when concepts or decisions are resolved.

## Use the glossary vocabulary

Use the terms defined in `CONTEXT.md` in specifications, issue titles, tests, implementation plans, and code-facing domain interfaces. Do not substitute terms explicitly listed under `_Avoid_`.

If a required concept is absent, reconsider whether the implementation is inventing unnecessary language. Record a real gap through `/domain-modeling`.

## Flag ADR conflicts

Surface any proposed change that contradicts an accepted ADR. Do not silently override an architectural decision.
