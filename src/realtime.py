import argparse, os, sys, time
from collections import defaultdict, deque
import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import ByteTrack
from scorenet import ScoreNet
from kimore_data import USE_JOINTS, COCO, H36M, normalize_h36m

COCO_SEL = [COCO.index(j) for j in USE_JOINTS]
LH, RH = USE_JOINTS.index('left_hip'), USE_JOINTS.index('right_hip')
LS, RS = USE_JOINTS.index('left_shoulder'), USE_JOINTS.index('right_shoulder')

def normalize_live(win):
    root = (win[:, LH] + win[:, RH]) / 2.0
    win = win - root[:, None, :]
    sh = (win[:, LS] + win[:, RS]) / 2.0
    scale = np.linalg.norm(sh, axis=-1).mean()
    return win / (scale + 1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0', help='webcam index, video path, or RTSP url')
    ap.add_argument('--ckpt', default='outputs/fold0_TS.pt')
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--exercise', type=int, default=1)
    ap.add_argument('--every', type=int, default=8, help='score every N frames')
    ap.add_argument('--save', default='', help='optional annotated output video')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--lift', action='store_true',
                    help='lift 2D->3D with MotionAGFormer (needs an h36m17 checkpoint)')
    args = ap.parse_args()

    from rtmlib import RTMO
    pose = RTMO('rtmo-m.onnx', model_input_size=(640, 640),
                backend='onnxruntime', device='cuda')
    tracker = ByteTrack()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    layout = ck['args'].get('layout', 'coco13')
    if args.lift and layout != 'h36m17':
        sys.exit("--lift needs a checkpoint trained with --layout h36m17 "
                 f"(this one is '{layout}'); otherwise train and deploy disagree.")
    net = ScoreNet(dual=ck['args'].get('dual', False),
                   n_joints=17 if layout == 'h36m17' else 13).to(dev).eval()
    net.load_state_dict(ck['model'])
    mu, sd = ck['norm']

    lifter = None
    if args.lift:
        from lifter import Lifter
        lifter = Lifter(device=dev)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        hint = ''
        if isinstance(src, str) and '?listen' in src:
            hint = ("\n  OpenCV cannot listen on a TCP port -- VideoCapture only dials out.\n"
                    "  Run  ./stream_listen.sh 9999  in another shell (it listens via\n"
                    "  ffmpeg and relays into a FIFO), then use:\n"
                    "    --source /tmp/rehab_stream.ts\n"
                    "  See README section 'Using your laptop webcam'.")
        elif isinstance(src, int):
            import glob as _g
            if not _g.glob('/dev/video*'):
                hint = ("\n  This machine has no camera (no /dev/video*). A webcam index only\n"
                        "  works on the machine physically attached to the camera.\n"
                        "  Options:\n"
                        "    - stream from your laptop:  ./stream_listen.sh 9999\n"
                        "                                then --source /tmp/rehab_stream.ts\n"
                        "    - or an MJPEG URL:          --source http://<phone-ip>:8080/video\n"
                        "    - or a recorded file:       --source clip.mp4\n"
                        "  See README section 'Using your laptop webcam'.")
        sys.exit(f'cannot open source: {args.source}{hint}')

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    hist = defaultdict(lambda: deque(maxlen=args.window))
    conf_hist = defaultdict(lambda: deque(maxlen=args.window))
    smooth = defaultdict(lambda: deque(maxlen=5))
    ex_t = torch.tensor([args.exercise], device=dev)
    n, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        kpts, scores = pose(frame)
        kpts, scores = np.array(kpts), np.array(scores)

        tracks = tracker.update(kpts, scores) if len(kpts) else []
        for t in tracks:

            hist[t.id].append(t.kpt if lifter else t.kpt[COCO_SEL])
            conf_hist[t.id].append(t.score_vec if lifter else None)

            if len(hist[t.id]) == args.window and n % args.every == 0:
                win = np.stack(hist[t.id]).astype(np.float32)
                if lifter:
                    cf = np.stack(conf_hist[t.id]).astype(np.float32)
                    h, w = frame.shape[:2]
                    win = lifter.lift(win, cf, w, h)
                    win = normalize_h36m(win)
                else:
                    win = normalize_live(win)
                    win = np.concatenate([win, np.zeros((*win.shape[:2], 1), np.float32)], -1)
                with torch.no_grad():
                    p = net(torch.from_numpy(win.astype(np.float32))[None].to(dev), ex_t)
                smooth[t.id].append(float(p.item() * sd + mu))

            x1, y1, x2, y2 = t.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lbl = f'ID{t.id}'
            if smooth[t.id]:
                lbl += f'  score {np.mean(smooth[t.id]):.1f}'
            cv2.putText(frame, lbl, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for x, y in t.kpt[COCO_SEL]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

        n += 1
        if writer:
            writer.write(frame)
        if not args.headless:
            cv2.imshow('rehab', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    print(f'{n} frames, {n / (time.time() - t0):.1f} FPS end-to-end')

if __name__ == '__main__':
    main()
