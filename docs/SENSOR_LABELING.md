# Sensor labeling

Learning V2 observations carry optional **confidence** per sensor (`player_hp`, `has_target`, `in_combat`, …). Heuristic vision starts with estimated confidence until labeled metrics exist.

## Thresholds

Configure under `learning_v2.sensor_thresholds` (values in `[0, 1]`):

```json
"learning_v2": {
  "sensor_thresholds": {
    "player_hp": 0.40,
    "target_hp": 0.40,
    "has_target": 0.50,
    "in_combat": 0.50,
    "motion": 0.30,
    "hostile_count": 0.40
  },
  "confidence_threshold": 0.45
}
```

- Readings below a sensor threshold should be treated as **unknown** rather than trusted False/0.5 defaults
- Policy `confidence_threshold` gates HybridPolicy: low BC confidence → scripted fallback

## Labeling workflow (manual)

1. Capture frames: `PYTHONPATH=. python3 scripts/capture_once.py --ocr`
2. Inspect ROIs / OCR under `data/playmind/owned/latest.png`
3. Record ground-truth beside demos (`skill`, notes, or future label fields in `meta.jsonl`)
4. Keep `sensor_warnings` in diagnostics when sensors disagree — export via [MIGRATION.md](./MIGRATION.md) diagnostics section / `scripts/export_diagnostics.py`

Automated labeling UI is a follow-on; until then prefer conservative thresholds and scripted fallbacks.

See also: [docs/LEARNING_ARCHITECTURE_V2.md](./LEARNING_ARCHITECTURE_V2.md)
