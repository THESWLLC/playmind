# Offline video import

Video import probes a local file with `ffprobe`, computes SHA-256, records media
metadata and provenance, then copies or references the source in a Studio
project. It never captures a screen or contacts a game client.

## Current state

- Import/probe/project storage: implemented and unit-tested at the failure and
  persistence boundaries.
- FFmpeg frame extraction: implemented; actual codecs remain host-dependent.
- GUI path-form import and extraction actions: implemented. File picker,
  progress reporting, and playable timeline are not.
- `change_aware`: milestone stub using half-interval uniform samples, not visual
  change ranking.
- `keyframes`: implemented with FFmpeg, but extracted timestamps are currently
  `null`.

## Install FFmpeg

```powershell
winget install Gyan.FFmpeg
```

```bash
# Debian/Ubuntu/WSL
sudo apt update && sudo apt install -y ffmpeg
```

```bash
# macOS
brew install ffmpeg
```

Open a new shell, then verify both tools:

```bash
ffmpeg -version
ffprobe -version
```

## Supported input and stored metadata

Extensions: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`.

The importer records the resolved original/stored paths, storage mode, SHA-256,
duration, dimensions, FPS, audio presence, format, and byte size. A valid
extension without a video stream is rejected. A failed probe creates no
project.

## Import

Start the GUI with `python scripts/start_studio.py` or
`start_playmind_studio.bat`. In **Projects/Import**, enter the local path,
choose copy/reference, confirm rights/local ML use, and click **Import**.
There is no dedicated import CLI flag; equivalent backend:

```bash
python - <<'PY'
from playmind.studio.video_import import import_video
from playmind.studio.provenance import ProvenanceRecord

project = import_video(
    "recordings/session.mp4",
    provenance=ProvenanceRecord(
        source_type="user_owned_recording",
        source_id="session-001",
        rights_confirmed=True,
        training_use_allowed=True,
        private_use_only=True,
        notes="Recorded and retained by me.",
    ),
    name="Session 001",
    mode="copy",
)
print(project)
PY
```

### Copy versus reference

| Mode | Behavior | Use when |
|---|---|---|
| `copy` | Copies media into the project's `source/` directory | Default; project should remain self-contained |
| `reference` | Stores the absolute original path | File is large and its location/retention is controlled |

Reference mode does not protect against a moved, deleted, or replaced file.
The import-time SHA-256 remains recorded, but extraction currently opens the
stored path and does not re-verify its hash automatically.

## Extract frames

```bash
python - <<'PY'
from playmind.studio.frame_extractor import extract_frames

PROJECT_ID = "<project-id>"
print(extract_frames(
    PROJECT_ID,
    strategy="overview",
    interval_seconds=10.0,
))
PY
```

Available source-level options:

| Argument | Values/default | Meaning |
|---|---|---|
| `strategy` | `overview` (default), `change_aware`, `keyframes`, `manual` | Sampling method |
| `interval_seconds` | `10.0` | Uniform/manual spacing; must be positive |
| `ranges` | iterable of `[start, end]` | Required by `manual` unless explicit timestamps are given |
| `timestamps` | iterable of seconds | Exact non-negative positions; overrides strategy selection |
| `ffmpeg` | path or `None` | Optional executable override |

The GUI currently exposes `overview`, `uniform`, and `scene_change`, but only
`overview` matches the backend. `uniform` and `scene_change` currently fail
with “unknown extraction strategy.” Use the backend for `change_aware`,
`keyframes`, or `manual` until the UI names are corrected.

Manual example:

```bash
python - <<'PY'
from playmind.studio.frame_extractor import extract_frames
print(extract_frames(
    "<project-id>",
    strategy="manual",
    ranges=[(12.0, 20.0), (75.0, 80.0)],
    interval_seconds=2.0,
))
PY
```

Each run creates a new immutable-style frame-set directory such as
`frames/overview-001/` and a `frames.json` manifest. Re-running does not delete
previous sets.

## Analyze extracted frames

```bash
python - <<'PY'
from playmind.studio.offline_analysis import analyze_project
print(analyze_project("<project-id>", do_ocr=False))
PY
```

Analysis uses existing still-image heuristics for health, objective text,
death/ghost state, and target state. Results start as `suggested`. OCR requires
the local OCR stack and can be enabled with `do_ocr=True`. This is heuristic
analysis, not ground truth; review it before export.

## Project layout

```text
data/playmind/studio/projects/<project-id>/
├── project.json
├── annotations.json
├── analysis.json
├── source/
├── frames/<frame-set-id>/frames.json
└── exports/
```

The default project ID is the first 16 characters of the source SHA-256.
Collisions/reimports receive `-2`, `-3`, and so on.

## Failure recovery

- `ffmpeg`/`ffprobe` missing: install both and restart the shell.
- Unsupported extension: remux to a supported container; renaming is not
  sufficient.
- “no video stream”: inspect the source with `ffprobe`.
- Reference path missing: restore it at the recorded path or import again.
- Extraction partial failure: keep the original project; run a new extraction
  after fixing the codec/tool issue. Review/remove partial frame directories
  manually if necessary.

See [Troubleshooting](./TROUBLESHOOTING.md) and
[Data Provenance and Permission](./DATA_PROVENANCE_AND_PERMISSION.md).
