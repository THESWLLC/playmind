# Feature schema v2

## Status

Schema v2 encoding, compatibility checks, dataset use, and train-only normalization are implemented and unit-tested.

```python
FEATURE_SCHEMA_VERSION = 2
```

The recurrent policy uses an ordered 54-D structured vector. The order is checkpoint metadata and is validated on load.

## Value, known, confidence

Eleven optional sensors are encoded as triples:

```text
<sensor>_value, <sensor>_known, <sensor>_confidence
```

They cover player/target HP, target/combat/death/ghost state, motion, nearby hostiles/count, blocking modal, and objective progress. The vector also includes always-known counters, life-phase one-hot fields, and temporal-summary fields.

For an unknown sensor:

```text
value=0, known=0, confidence=0
```

For a known false boolean:

```text
value=0, known=1, confidence=<sensor confidence or 1>
```

Therefore **unknown is not known-false**, even though their value slots are both zero. Consumers must retain the known/confidence channels.

## Normalization

`DemonstrationDataset.fit_normalizer()` is allowed only for `split="train"`. The resulting mean/std and exact feature names are stored in recurrent checkpoint metadata and reused for validation, test, and inference.

Known masks and life-phase one-hots are left unchanged. Other eligible numeric fields are z-normalized; padded timesteps are restored to zero after normalization.

## Compatibility and legacy data

- Recurrent checkpoints require feature schema 2, dimension 54, and an exact ordered-name match. Mismatches fail rather than silently remapping.
- Legacy `SkillPolicyV2` checkpoints use the schema-v1 38-D last-frame layout and must use the legacy loader/adapter.
- Demonstration files currently remain recording `schema_version=1`; the dataset converts their observation dictionaries to model feature schema v2 at load time. These are different version domains.
- Missing legacy fields remain unknown where the observation parser can preserve that fact. Do not prefill absent sensors with false/zero before encoding.

Primary symbols: `FEATURE_SCHEMA_VERSION`, `FEATURE_NAMES`, `structured_feature_vector_v2`, and `FeatureNormalizer` in `playmind/models/feature_schema.py`.
