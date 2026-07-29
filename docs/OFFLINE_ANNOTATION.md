# Offline annotation

Offline annotations mark time ranges in imported recordings as skills, goals,
or outcomes. The storage/review API and browser time-range review panel are
implemented. F7/F8 review hotkeys are optional and focus-gated. A playable
video timeline, visual undo history, and durable undo history have not landed.

## Review rule

Every imported analysis result, transcript suggestion, and new timeline
segment starts as `suggested`. Only a human-reviewed segment may become
training-eligible. `rejected`, `unknown`, and `unusable` segments are excluded.

Review statuses:

- `suggested`: machine/import/user draft; not eligible
- `reviewed`: human accepted; eligible unless category is excluded
- `rejected`: human rejected; never eligible

Segment types are `skill`, `goal`, and `outcome`. The Studio dataset bridge
currently exports only reviewed `skill` segments to planner SFT.

## Add and review a segment

In **Annotation timeline**, enter start/end seconds, type, and category; click
**Add**, select the row, then accept/reject. When `pynput` is available, F7
accepts and F8 rejects only while the review panel is focused. Buttons remain
available without `pynput`.

```bash
python - <<'PY'
from playmind.studio.annotations import AnnotationStore, TimelineSegment

store = AnnotationStore("<project-id>")
segment = store.add(TimelineSegment(
    start=12.5,
    end=18.0,
    segment_type="skill",
    category="recover_health",
    label="Recover after low health",
    notes="Health rises; no combat overlap.",
))
reviewed = store.review(segment.segment_id, accepted=True)
print(reviewed.to_dict())
PY
```

Times are seconds and must satisfy `0 <= start <= end`. Categories must be an
implemented PlayMind skill or one of `success`, `failure`, `unknown`, and
`unusable`.

List valid categories:

```bash
python - <<'PY'
from playmind.studio.annotations import annotation_categories
print("\n".join(sorted(annotation_categories())))
PY
```

## Update, reject, remove, undo

```python
from playmind.studio.annotations import AnnotationStore

store = AnnotationStore("<project-id>")
store.update("<segment-id>", start=13.0, end=17.5, notes="Adjusted boundaries")
store.review("<segment-id>", accepted=False)
store.remove("<segment-id>")
store.undo()
```

Undo is process-local: `AnnotationStore` keeps prior snapshots only for changes
made through that instance. Closing the process loses the undo stack, although
saved annotations remain. This is not a durable revision history.

## Human labeling procedure

1. Watch enough context before and after the range.
2. Mark the smallest range that supports one unambiguous skill/outcome.
3. Use observable facts in notes; do not infer hidden game state.
4. Use `unknown` when evidence is ambiguous and `unusable` for corrupt,
   obstructed, irrelevant, or contaminated material.
5. Check lifecycle boundaries carefully: death, release, ghost runback,
   loading, modal UI, and recovery should not be merged into ordinary combat.
6. Have a second reviewer inspect benchmark examples and disputed labels.
7. Keep all clips from one project/source hash in one data split.

Overlapping segments are currently allowed. Resolve contradictory overlap
manually before export.

## Transcript-assisted suggestions

Supported local transcript files are `.srt`, `.vtt`, and `.txt`:

```bash
python - <<'PY'
from playmind.studio.app import StudioApp

app = StudioApp()
app.select_project("<project-id>")
app.import_transcript("recordings/session.srt")
print(app.transcript_suggestions())
PY
```

Suggestions use keyword rules, not ASR or an LLM, and always contain
`training_eligible: false`. They are not automatically added to annotations.
The reviewer must check timing, create the segment, and accept it explicitly.
TXT lines receive timestamp `0.0` with no end and therefore need manual timing.

## Review offline analysis

`analysis.json` contains suggested sensor detections. There is not yet an
analysis-review API or GUI on this branch. If reviewed visual-state export is
required, review and edit the records conservatively so each accepted record
has `review_status: "reviewed"`; preserve its `source_frame`, `name`, and
timestamp. Back up the project first because manual JSON editing bypasses
validation.

## Export eligibility

A segment is exported only when:

- its project provenance is training-eligible;
- `review_status` is `reviewed`;
- category is neither `unknown` nor `unusable`; and
- segment type is `skill` for planner SFT.

Run export and inspect counts:

```bash
python - <<'PY'
import json
from playmind.studio.dataset_bridge import export_reviewed_projects
result = export_reviewed_projects(["<project-id>"])
print(json.dumps(result, indent=2))
PY
```

Zero counts are a review/provenance signal, not a reason to bypass the gates.
