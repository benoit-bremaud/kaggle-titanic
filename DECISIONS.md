# Project-Specific Decision Log

This file records architectural decisions specific to **this competition/project**.
For global decisions applying to all Kaggle projects, see the [global DECISIONS.md](../DECISIONS.md).

---

<!-- Template for new decisions:

## ADR-P001: Decision title

**Status:** Proposed | Accepted | Deprecated | Superseded
**Date:** YYYY-MM-DD

**Context:** Why is this decision needed?

**Options considered:**

| Option | Pros | Cons |
| --- | --- | --- |
| Option A | ... | ... |
| Option B | ... | ... |

**Decision:** What was decided.

**Consequences:** What changes as a result.

-->

## Global rules applied to this project

All best practices from [global DECISIONS.md](../DECISIONS.md) apply here, specifically:

- **ADR-016** — CI with ruff only (no SonarQube)
- **ADR-017** — pytest on `src/` only
- **ADR-018** — KISS and YAGNI, not SOLID
- **ADR-019** — Reproducibility, DRY, no data leakage
- **ADR-020** — Pipeline-centric ML approach
- **ADR-021** — Stratified K-Fold, local CV > public leaderboard
- **ADR-022** — Notebook hygiene (top-to-bottom, Restart & Run All)
- **ADR-023** — Raw data immutable and gitignored

No Titanic-specific decisions yet. They will be added as the project progresses.
