from __future__ import annotations

import argparse, json, os, socket, socketserver, struct, sys, threading, time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import ByteTrack
from scorenet import ScoreNet
from kimore_data import USE_JOINTS, COCO, normalize_h36m

COCO_SEL = [COCO.index(j) for j in USE_JOINTS]
LH, RH = USE_JOINTS.index('left_hip'), USE_JOINTS.index('right_hip')
LS, RS = USE_JOINTS.index('left_shoulder'), USE_JOINTS.index('right_shoulder')

STATE: dict = {}

def recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf

def normalize_live(win):
    root = (win[:, LH] + win[:, RH]) / 2.0
    win = win - root[:, None, :]
    sh = (win[:, LS] + win[:, RS]) / 2.0
    return win / (np.linalg.norm(sh, axis=-1).mean() + 1e-6)

class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"[+] client {peer}", flush=True)

        tracker = ByteTrack()
        from collections import defaultdict, deque
        hist = defaultdict(lambda: deque(maxlen=STATE["window"]))
        conf_hist = defaultdict(lambda: deque(maxlen=STATE["window"]))
        smooth = defaultdict(lambda: deque(maxlen=5))
        n = 0

        try:
            while True:
                head = recv_exactly(self.request, 4)
                if head is None:
                    break
                (size,) = struct.unpack(">I", head)
                if not (0 < size <= 8 << 20):
                    break
                payload = recv_exactly(self.request, size)
                if payload is None:
                    break

                t0 = time.time()
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    err = json.dumps({"people": [], "ms": 0.0, "n": n,
                                      "error": f"could not decode {len(payload)} bytes as JPEG"}).encode()
                    self.request.sendall(struct.pack(">I", len(err)) + err)
                    continue

                h, w = frame.shape[:2]
                with STATE["lock"]:
                    kpts, scores = STATE["pose"](frame)
                kpts, scores = np.array(kpts), np.array(scores)
                tracks = tracker.update(kpts, scores) if len(kpts) else []

                people = []
                for t in tracks:
                    hist[t.id].append(t.kpt if STATE["lifter"] else t.kpt[COCO_SEL])
                    if STATE["lifter"] is not None:
                        conf_hist[t.id].append(t.score_vec)

                    if len(hist[t.id]) == STATE["window"] and n % STATE["every"] == 0:
                        win = np.stack(hist[t.id]).astype(np.float32)
                        if STATE["lifter"] is not None:
                            cf = np.stack(conf_hist[t.id]).astype(np.float32)
                            win = normalize_h36m(STATE["lifter"].lift(win, cf, w, h))
                        else:
                            win = normalize_live(win)
                            win = np.concatenate([win, np.zeros((*win.shape[:2], 1), np.float32)], -1)
                        with torch.no_grad(), STATE["lock"]:
                            p = STATE["net"](
                                torch.from_numpy(win.astype(np.float32))[None].to(STATE["device"]),
                                STATE["ex"])
                        smooth[t.id].append(float(p.item() * STATE["sd"] + STATE["mu"]))

                    x1, y1, x2, y2 = (t.box / np.array([w, h, w, h])).tolist()
                    people.append({
                        "id": int(t.id),
                        "box": [x1, y1, x2, y2],
                        "kp": [[float(x) / w, float(y) / h, float(s)]
                               for (x, y), s in zip(t.kpt, t.score_vec)],
                        "score": (float(np.mean(smooth[t.id])) if smooth[t.id] else None),
                        "ready": len(hist[t.id]) == STATE["window"],
                        "fill": len(hist[t.id]) / STATE["window"],
                    })

                n += 1
                reply = json.dumps({"people": people, "ms": (time.time() - t0) * 1000,
                                    "n": n, "exercise": STATE["ex_id"]}).encode()
                self.request.sendall(struct.pack(">I", len(reply)) + reply)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            print(f"[-] client {peer} disconnected after {n} frames", flush=True)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

def main() -> None:
    p = argparse.ArgumentParser(description="Rehabilitation quality-score inference server.")
    p.add_argument("--ckpt", default="outputs/fold0_TS.pt")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 is correct when the client comes through an SSH tunnel")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--exercise", type=int, default=1, help="which KiMoRe exercise (1-5)")
    p.add_argument("--every", type=int, default=4, help="score every N frames")
    p.add_argument("--lift", action="store_true",
                   help="MotionAGFormer 2D->3D (needs an h36m17 checkpoint)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    layout = ck["args"].get("layout", "coco13")
    if args.lift and layout != "h36m17":
        sys.exit(f"--lift needs a checkpoint trained with --layout h36m17 (this is '{layout}')")

    net = ScoreNet(dual=ck["args"].get("dual", False),
                   n_joints=17 if layout == "h36m17" else 13).to(dev).eval()
    net.load_state_dict(ck["model"])
    mu, sd = ck["norm"]

    from rtmlib import RTMO
    pose = RTMO("rtmo-m.onnx", model_input_size=(640, 640),
                backend="onnxruntime", device=args.device)

    lifter = None
    if args.lift:
        from lifter import Lifter
        lifter = Lifter(device=args.device)

    STATE.update(net=net, pose=pose, lifter=lifter, device=dev, mu=mu, sd=sd,
                 window=ck["args"].get("window", 64), every=args.every,
                 ex=torch.tensor([args.exercise], device=dev), ex_id=args.exercise,
                 lock=threading.Lock())

    print(f"checkpoint {args.ckpt} | target {ck['args'].get('target')} | layout {layout}")
    print(f"window {STATE['window']} frames | scoring every {args.every} | "
          f"3D lift {'on' if lifter else 'off'} | device {dev}")
    print(f"listening on {args.host}:{args.port}")
    print(f"tunnel from the laptop with:  ssh -N -L {args.port}:localhost:{args.port} "
          f"<user>@<server>")
    Server((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
