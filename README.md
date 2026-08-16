# Rehab Quality Scoring Pipeline

`RGB → RTMO (detect+pose) → ByteTrack → MotionAGFormer (2D→3D) → ScoreNet → quality score`

Isolated conda env **`rehab`** (does not touch your existing envs).

## Quick start

**Live scoring from your laptop webcam** 

```bash
# server (this GPU box)
conda activate rehab && cd ~/rehab_pipeline
python -u src/serve.py --ckpt outputs/fold0_TS.pt --port 5556 --exercise 2
```
```bash
# laptop
python client_cam.py --host localhost --port 5556 --source 1
```

**Training** (subject-disjoint 5-fold CV):
```bash
conda activate rehab && cd ~/rehab_pipeline
python src/train.py --target TS --epochs 40 --folds 5
```

**Scoring a recorded file on the server** (no camera involved):
```bash
python src/realtime.py --source clip.mp4 --ckpt outputs/fold0_TS.pt \
    --headless --save outputs/out.mp4
```
`--headless` is required here — the server has no display. Copy the annotated
result back with `scp`.

**3D path** — train on the H36M-17 layout, then lift at inference:
```bash
python src/train.py --target TS --epochs 40 --folds 5 --layout h36m17 --out outputs_h36m
python src/serve.py --ckpt outputs_h36m/fold0_TS.pt --port 5556 --lift
```

## Live laptop webcam (client/server) — recommended

The laptop captures and draws; the server does all inference. The laptop needs
only `opencv-python` and `numpy` — no torch, no model downloads, no CUDA.

**1. Server** (this GPU box):
```bash
conda activate rehab && cd ~/rehab_pipeline
python -u src/serve.py --ckpt outputs/fold0_TS.pt --port 5556 --exercise 5
```

**2. Client** (laptop):
```bash
pip install -r requirements-client.txt      # opencv-python, numpy
python client_cam.py --host localhost --port 5556 --source 1
```

A window opens with your camera, one coloured box + skeleton per person, track
IDs and live scores. **Press `q` or `Esc` to quit.**

Display size is independent of what gets sent to the server:

| Flag | Effect |
|---|---|
| `--width 1600` | window width in pixels (default 1280; `0` = camera's native size) |
| `--fullscreen` | fill the screen |
| `--capture-width 1920` | ask the camera for a higher capture resolution |
| `--send-width 640` | width sent to the server — affects pose accuracy and bandwidth, not the window |

The window is resizable by dragging regardless; `--width` sets its initial size
and scales the drawn frame to match.


### Choosing the exercise

`--exercise N` selects which KiMoRe movement the score head conditions on. It
must match what the person is actually doing or the number is meaningless:

| N | Movement |
|---|---|
| 1 | Lifting of the arms |
| 2 | Lateral tilt of the trunk with arms extended |
| 3 | Trunk rotation |
| 4 | Pelvis rotation on the transverse plane |
| 5 | Squatting |


### Troubleshooting

```bash
python client_cam.py --list-cameras     # which index is your webcam
python client_cam.py --probe            # camera + connection + round trip
```


### Interpreting the score

The head was trained on KiMoRe's Kinect-derived skeletons of single,
front-facing, full-body subjects at a fixed distance. Absolute values on a
webcam are not calibrated to the clinical scale; relative differences between
repetitions are more trustworthy than the raw number. Cross-validated
correlation on unseen subjects is ~0.6 (see below).

## Files
| Path | Role |
|---|---|
| `src/serve.py` | TCP inference server (GPU side) for the laptop client |
| `client_cam.py` | self-contained laptop webcam client (no torch needed) |
| `requirements-client.txt` | laptop deps: `opencv-python`, `numpy` — no torch |
| `src/kimore_data.py` | KiMoRe loader, Kinect-25 → COCO-13 retarget, normalization, label parsing |
| `src/scorenet.py` | ST-GCN `ScoreNet` (0.28M params), exercise-conditioned |
| `src/lifter.py` | MotionAGFormer-S 2D→3D lifting, COCO-17 → H36M-17 remap |
| `src/tracker.py` | ByteTrack multi-person association |
| `src/train.py` | Subject-wise `GroupKFold` training + metrics |
| `src/realtime.py` | Local inference on a file (server-side, no camera) |
| `models/end2end.onnx` | RTMO-m weights (also cached at `~/.cache/rtmlib/`) |
| `models/motionagformer-s-h36m.pth.tr` | MotionAGFormer-S checkpoint (4.8M params) |
| `models/MotionAGFormer/` | upstream repo (model definition) |

## Measured on this machine
- RTMO-m @720p: **173 FPS** (GPU)
- MotionAGFormer-S lift: **33 FPS** (81-frame window)
- End-to-end 2D (pose+track+score): **65.8 FPS**
- End-to-end 3D (pose+track+lift+score): **67.2 FPS**
- Client/server (local loopback): **98 FPS, 7 ms server latency/frame**
- Dataset: 8,613 windows / 77 subjects / 383 labeled (subject,exercise) pairs
- Full 5-fold subject-disjoint CV (40 epochs):

| Target | MAE | Pearson | Spearman |
|---|---|---|---|
| TS (total score) | 6.08 ± 0.74 | **0.596 ± 0.047** | 0.573 ± 0.046 |
| PO (postural) | 2.44 ± 0.08 | 0.578 ± 0.041 | 0.524 ± 0.042 |
| CF (clinical factors) | 4.36 ± 0.68 | 0.531 ± 0.074 | 0.518 ± 0.048 |
| TS dual-stream (pos+quat) | 6.17 ± 0.65 | 0.593 ± 0.036 | 0.550 ± 0.026 |
| TS h36m17 (3D deploy path) | 6.24 ± 0.59 | 0.579 ± 0.042 | 0.543 ± 0.047 |

