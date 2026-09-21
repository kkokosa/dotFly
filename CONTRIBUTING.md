# Contributing

Thanks for looking. A few rules keep this project honest and fast:

- **Open an issue first** for anything beyond a typo; the roadmap features (F1–F4 in `PLAN.md`
  §11, `docs/roadmap.md`) are the intended directions.
- **Pull requests only**, against `main`; CI (build + tests on Ubuntu and Windows) must pass; the
  maintainer reviews every change. Squash or rebase merges.
- **Honesty in wording.** Never describe the simulation as a digital fly, an upload, or a brain that
  "plays" a game. The wiring is measured; everything else is *calibrated*, *fixed* or *trained* and
  must be labelled so wherever it appears (HUD, docs, code comments).
- **Determinism and the hot path.** Same checkpoint + options + seed ⇒ identical spikes at any
  thread count; no allocations on the step path; every optimisation comes with `dotfly bench`
  numbers. Read `CLAUDE.md` for the rules and `PLAN.md` §12 for the findings before touching the
  kernel or the Brian2 semantics.
- **Data is never committed.** Connectome releases stay in the git-ignored `.data/`; fixtures carry
  their attribution.
- **Tests**: unit tests are data-free and fast; anything that needs a real checkpoint is an
  integration test that skips when `.data/` is absent.
- Licence: GPL-3.0. By contributing you agree your contribution is licensed the same way.
