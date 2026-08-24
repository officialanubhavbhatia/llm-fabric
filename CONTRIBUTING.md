# Contributing

Authorship of production code in this repository is **Anubhav Bhatia**. Do not
add AI, Cursor, or tool co-author trailers.

## Before you change architecture

Read `docs/constitution.md`. It overrides convention and inference.
`ARCHITECTURE.md` must stay consistent with it and must distinguish what is
built from what is proposed.

## Local checks

```text
make check
```

That runs lint, types, and the test suite. Cross-tenant leaks have their own
gate:

```text
make test-isolation
```

Evaluation regressions:

```text
make eval-gate
```

Do not use `SKIP_EVALS` or `LLM_FABRIC_SKIP_EVALS`. Those variables are refused
by `llm-fabric-eval gate`. A failed gate is overridden only by an audited
administrative action.

## Security reports

See `SECURITY.md`. Do not file vulnerability details in public issues.

## License

See `LICENSE_DECISION.md`. A license has not been selected by the owner.
