# Box Tracking Standalone

Standalone test tool for top-view carton/fridge box tracking. It uses a YOLO model to validate detections in a configured ROI, while the tracking box is maintained mainly by prediction to reduce size jitter under occlusion, foam-board separation, and temporary detection loss.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the YOLO weight file and test video next to the script, or pass absolute paths with `--model` and `--source`.

## Run

The script contains the current default test command:

```powershell
python box_tracking_standalone.py
```

Equivalent explicit command:

```powershell
python box_tracking_standalone.py --model best_26m.pt --source record_20260211_104117.mkv --roi 1436 450 4333 1133 --class-id 1
```

By default, the display is cropped to the ROI with padding so the tracking effect is easier to inspect.

## Debug Log

To export per-frame tracking metrics:

```powershell
python box_tracking_standalone.py --debug-log tracking_debug.csv
```

The CSV contains frame index, track id, state, hit/miss count, stable size, visible ratio, center offset ratio, detection area, and reference area.

## Notes

Model weights and video recordings are intentionally ignored by git. Keep large runtime assets outside the repository or upload them as GitHub release assets if needed.
