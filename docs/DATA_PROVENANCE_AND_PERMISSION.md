# Data provenance and permission

Provenance records where a recording came from and whether its intended use is
allowed. The code applies conservative eligibility rules, but a checked field
is not legal proof. The operator must keep the underlying consent, license, and
retention evidence.

This document is operational guidance, not legal advice.

## Source types

| `source_type` | Minimum code gate for training |
|---|---|
| `user_owned_recording` | rights/permission or explicit `training_use_allowed`; a narrow private-only override exists but should not replace verification |
| `friend_provided` | rights/permission or training allowance **and** consent |
| `consenting_creator` | rights/permission or training allowance **and** consent |
| `licensed_dataset` | rights/permission or training allowance **and** license confirmation |
| `synthetic` | eligible by code; still verify asset/model licenses |
| `unknown` | never eligible |

Fields include `source_id`, `rights_confirmed`, `permission_confirmed`,
`consent_confirmed`, `license_confirmed`, `training_use_allowed`,
`private_use_only`, attribution, license name, source URI, notes, and metadata.

## Before import

Record answers to all of these:

1. Who created and owns the recording?
2. Who appears or speaks in it, and did they consent to this use?
3. Is local ML training allowed, or only viewing?
4. Are derived frames, labels, adapters, and metrics allowed?
5. Is redistribution allowed? If not, mark it private and do not publish
   project data or trained artifacts that may reproduce source material.
6. Is attribution required?
7. What is the retention/deletion requirement?
8. Does the source platform permit download and this use?
9. Does the game/software license permit recording and derivative analysis?
10. Is the intended use offline only?

Do not scrape Twitch, YouTube, public logs, or dataset mirrors merely because
they are publicly accessible. Public access is not training permission.

## Create a provenance record

```python
from playmind.studio.provenance import ProvenanceRecord

record = ProvenanceRecord(
    source_type="consenting_creator",
    source_id="creator-consent-2026-001",
    rights_confirmed=True,
    consent_confirmed=True,
    training_use_allowed=True,
    private_use_only=True,
    attribution="Creator display name, if required",
    notes="Written consent retained outside the repository.",
    metadata={
        "consent_record": "local://consent/creator-001.pdf",
        "retention_review_date": "2027-07-29",
    },
)
```

Avoid storing private contact information or consent documents in Git. Store a
stable reference in metadata and protect the source evidence separately.

Check the implemented rule:

```bash
python - <<'PY'
from playmind.studio.provenance import ProvenanceRecord, is_training_eligible
record = ProvenanceRecord(
    "user_owned_recording",
    rights_confirmed=True,
    training_use_allowed=True,
    private_use_only=True,
)
print(is_training_eligible(record))
PY
```

## Conservative defaults

- Use `unknown` if evidence is missing.
- Do not set a confirmation field based only on assumption.
- Prefer `copy` only when local retention is allowed.
- Do not use `allow_unverified_private=True` as a routine bypass. It only
  admits unverified user-owned recordings marked private; it does not establish
  rights.
- Keep source/project IDs stable so leakage detection can group related rows.
- Treat altered/re-encoded copies as the same source group.

## Dataset and split controls

The Studio export rejects ineligible projects and reports them in
`rejected_projects`. It groups examples with an episode ID derived from the
source SHA-256. Leakage detection flags a project ID, source ID, or source hash
appearing across multiple assigned splits when those fields are present.

The exporter's SFT/preference split logic is episode-aware, but operators must
also keep near-duplicate sessions, clips from the same recording, and revised
annotations in one split. A held-out benchmark must come from untouched
projects/sources.

## Publication checklist

Before committing, uploading, sharing, or registering an artifact:

- [ ] Source recording and extracted frames are excluded from Git.
- [ ] Consent/license permits the exact audience and purpose.
- [ ] Required attribution and notices are included.
- [ ] Personal data, chat, names, voices, and account identifiers are removed
      or authorized.
- [ ] Model base license permits adapter/merged/GGUF distribution.
- [ ] Benchmark examples do not redistribute protected content.
- [ ] Artifact is marked `live_use_prohibited` when derived from the protected
      offline profile.
- [ ] A deletion procedure can locate source, frames, exports, benchmarks, and
      derived model runs.

## Revocation or mistake

Stop training and sharing. Identify every project by `source_id`, SHA-256, and
project ID; quarantine source media, frames, exported rows, benchmark
scenarios, and derived runs. Document what was deleted and what cannot be
recalled. Do not silently flip provenance flags while retaining derived data.

See [Compliance Boundaries](./COMPLIANCE_BOUNDARIES.md) and
[Account Safety Architecture](./ACCOUNT_SAFETY_ARCHITECTURE.md).
