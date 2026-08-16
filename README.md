# Rehab Quality Scoring Pipeline

`RGB → RTMO (detect+pose) → ByteTrack → MotionAGFormer (2D→3D) → ScoreNet → quality score`

Isolated conda env **`rehab`** (does not touch your existing envs).

## Quick start

**Live scoring from your laptop webcam** — the normal way to run this. See
[Live laptop webcam](#live-laptop-webcam-clientserver--recommended) for detail.

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
`--lift` requires an `h36m17` checkpoint; using a `coco13` one exits with an
error rather than silently scoring a mismatched skeleton.

> `--source 0` does not work on the server: it has no camera (`/dev/video*` is
> empty). A camera index only resolves on the machine the camera is attached to,
> which is why the client/server split exists.

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

### Reaching the server

`serve.py` binds `127.0.0.1` by default, so the client connects through an SSH
tunnel:
```bash
ssh -N -L 5556:localhost:5556 <user>@<server>
```
If your SSH session already forwards the port, `--host localhost` just works
with no separate tunnel command.

On Windows, `bind: Permission denied` means the local port sits inside a
reserved range (Hyper-V/WSL). Check with
`netsh interface ipv4 show excludedportrange protocol=tcp` and pick a port
outside it, or map a different local port: `-L 3333:localhost:5556` then
`--port 3333`.

To skip the tunnel entirely, start the server with `--host 0.0.0.0` and connect
to the machine's address directly. That exposes the port to anyone who can reach
the host and sends webcam frames unencrypted — fine on a trusted lab network,
not something to leave running unattended.

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

### Reading the display

Each track shows `warming NN%` until its 64-frame window fills (~6 s at 10 Hz),
then switches to a score. Stuck at `warming` means the person is not being
tracked consistently — usually too close, partly out of frame, or poor lighting.

### Troubleshooting

```bash
python client_cam.py --list-cameras     # which index is your webcam
python client_cam.py --probe            # camera + connection + round trip
```

Use `rehab_pipeline/client_cam.py`, not the client from `utkinect_mlp` — they
share the wire format but not the reply schema (`people` here vs `top` there),
so mixing them raises a `KeyError`.

Measured with client and server on the same machine: **98 fps, 7 ms server
latency per frame**. Over a real tunnel the uplink dominates — tune with
`--hz`, `--quality`, `--send-width`.

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

## Why the numbers look "low"
Splits are **subject-disjoint**. Published KiMoRe results of 0.95+ Spearman
typically use random clip splits, which leak subject identity — the same person
appears in train and test. A leakage-controlled score near 0.6–0.8 is the
honest signal for unseen patients. Never switch to a random split to get a
prettier number.

## MotionAGFormer (integrated)

`--lift` runs MotionAGFormer-S on each track's 2D window and scores the metric
3D output. Joint layouts are the subtle part: RTMO emits COCO-17,
MotionAGFormer expects/returns Human3.6M-17, and KiMoRe is Kinect-25. The
`h36m17` layout builds H36M-17 straight from Kinect so the head trains on the
same skeleton it sees at inference.

## Notes
- `onnxruntime-gpu` needs CUDA 12 libs from torch; the env activation hook
  (`etc/conda/activate.d/cuda_libs.sh`) sets `LD_LIBRARY_PATH`. Without it
  inference silently drops to CPU (~4 FPS).
- KiMoRe ships **78** subjects; **77** train. `NE_ID2`'s ClinicalAssessment
  files are stubs — blank scores, and the `Subject ID` cell reads `E_ID1` — so
  it has no regression target and the loader drops it.
- Two subjects have `Es6` (no clinical score) — skipped by the loader.
- `NE_ID11/Es5` lacks `JointPosition` — skipped.
