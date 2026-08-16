from __future__ import annotations

import argparse, json, socket, struct, threading, time

import cv2
import numpy as np

def recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf

class Link:
    timeout = 15.0

    def __init__(self, host, port, quality, send_w):
        self.addr = (host, port)
        self.quality, self.send_w = quality, send_w
        self.sock: socket.socket | None = None
        self.people: list = []
        self.server_ms = self.rtt_ms = 0.0
        self.sent = 0
        self.error = ""
        self.lock = threading.Lock()

    def connect(self):
        s = socket.create_connection(self.addr, timeout=self.timeout)
        s.settimeout(self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        with self.lock:
            self.error = ""

    def send_frame(self, frame):
        if self.sock is None:
            self.connect()
        h, w = frame.shape[:2]
        if w > self.send_w:
            frame = cv2.resize(frame, (self.send_w, int(round(h * self.send_w / w))))
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        data = jpg.tobytes()
        t0 = time.time()
        self.sock.sendall(struct.pack(">I", len(data)) + data)
        head = recv_exactly(self.sock, 4)
        if head is None:
            raise ConnectionError("server closed the connection")
        (n,) = struct.unpack(">I", head)
        payload = recv_exactly(self.sock, n)
        if payload is None:
            raise ConnectionError("truncated reply")
        msg = json.loads(payload.decode())
        if msg.get("error"):
            raise ConnectionError(f"server: {msg['error']}")
        with self.lock:
            self.people = msg.get("people", [])
            self.server_ms = msg["ms"]
            self.rtt_ms = (time.time() - t0) * 1000
            self.sent += 1

    def snapshot(self):
        with self.lock:
            return list(self.people), self.server_ms, self.rtt_ms, self.sent, self.error

    def fail(self, e):
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.sock = None
        with self.lock:
            self.error = str(e)[:60]

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
KP_MIN = 0.3
PALETTE = [(0, 235, 120), (255, 190, 0), (0, 170, 255), (255, 105, 180), (180, 180, 60)]

def draw(frame, people, server_ms, rtt_ms, sent, err, hz):
    h, w = frame.shape[:2]
    for p in people:
        col = PALETTE[p["id"] % len(PALETTE)]
        x1, y1, x2, y2 = [int(v * s) for v, s in zip(p["box"], (w, h, w, h))]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        pts = [(int(x * w), int(y * h), s) for x, y, s in p["kp"]]
        for a, b in BONES:
            if a < len(pts) and b < len(pts) and pts[a][2] > KP_MIN and pts[b][2] > KP_MIN:
                cv2.line(frame, pts[a][:2], pts[b][:2], col, 2, cv2.LINE_AA)
        for x, y, s in pts:
            if s > KP_MIN:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        if p["score"] is not None:
            lab = f"ID{p['id']}  {p['score']:.1f}"
        else:
            lab = f"ID{p['id']}  warming {p['fill']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), col, -1)
        cv2.putText(frame, lab, (x1 + 4, max(th, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    status = (f"{len(people)} tracked | server {server_ms:5.1f} ms | rtt {rtt_ms:5.1f} ms | "
              f"sent {sent} @ {hz:.1f} Hz" if not err else f"DISCONNECTED: {err}")
    cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(frame, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255) if err else (210, 210, 210), 1, cv2.LINE_AA)
    return frame

BACKENDS = {"auto": 0, "dshow": getattr(cv2, "CAP_DSHOW", 700),
            "msmf": getattr(cv2, "CAP_MSMF", 1400), "v4l2": getattr(cv2, "CAP_V4L2", 200),
            "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", 1200)}

def open_camera(source, backend="auto"):
    if not source.isdigit():
        return cv2.VideoCapture(source)
    api = BACKENDS.get(backend, 0)
    return cv2.VideoCapture(int(source), api) if api else cv2.VideoCapture(int(source))

def list_cameras(args):
    print(f"probing indices 0..{args.max_index} with backend '{args.backend}'\n")
    found = 0
    for i in range(args.max_index + 1):
        cap = open_camera(str(i), args.backend)
        if not cap.isOpened():
            cap.release(); continue
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not ok or frame is None:
            print(f"  [{i}] opens but returns no frames (in use by another app?)"); continue
        found += 1
        cv2.imwrite(f"camera_{i}.jpg", frame)
        print(f"  [{i}] {w}x{h}  -> snapshot saved to camera_{i}.jpg")
    if not found:
        print("  no working cameras found")
        if args.backend == "auto":
            print("  on Windows try:  --backend dshow")

def probe(args):
    print("1. camera")
    cap = open_camera(args.source, args.backend)
    if not cap.isOpened():
        print(f"   FAIL: cannot open source {args.source!r}")
        print("   try --source 1, or --list-cameras; on macOS grant terminal camera access")
        return
    ok, frame = cap.read(); cap.release()
    if not ok or frame is None:
        print("   FAIL: opened the device but read() returned nothing"); return
    print(f"   OK: got a {frame.shape[1]}x{frame.shape[0]} frame")

    print("2. connection")
    link = Link(args.host, args.port, args.quality, args.send_width)
    try:
        link.connect()
    except OSError as e:
        print(f"   FAIL: {type(e).__name__}: {e}")
        print(f"   is the tunnel up?  ssh -N -L {args.port}:localhost:{args.port} <user>@<server>")
        return
    print(f"   OK: connected to {args.host}:{args.port}")

    print("3. round trip")
    try:
        link.send_frame(frame)
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}"); return
    people, ms, rtt, _, _ = link.snapshot()
    print(f"   OK: {len(people)} people detected, server {ms:.1f} ms, rtt {rtt:.1f} ms")
    if people:
        p = people[0]
        vis = sum(1 for _, _, s in p["kp"] if s > KP_MIN)
        print(f"   ID{p['id']}: {vis}/17 joints visible, window {p['fill']*100:.0f}% full")
    print("\nall stages passed -- run without --probe")

def main():
    p = argparse.ArgumentParser(description="Webcam client for the remote rehab scorer.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--source", default="0", help="camera index, or a video file to test with")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--send-width", type=int, default=640,
                   help="pose needs more detail than classification; 640 is a good default")
    p.add_argument("--backend", default="auto", choices=list(BACKENDS))
    p.add_argument("--mirror", action="store_true", help="flip horizontally for a selfie view")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--max-index", type=int, default=5)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()

    if args.list_cameras:
        list_cameras(args); return
    if args.probe:
        probe(args); return

    cap = open_camera(args.source, args.backend)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source!r} -- try --list-cameras")

    link = Link(args.host, args.port, args.quality, args.send_width)
    try:
        link.connect()
        print(f"connected to {args.host}:{args.port}")
    except OSError as e:
        raise SystemExit(f"cannot reach {args.host}:{args.port} -- is the SSH tunnel up? ({e})")

    latest = {"f": None}
    stop = threading.Event()

    def worker():
        nxt = time.time()
        while not stop.is_set():
            now = time.time()
            if now < nxt:
                time.sleep(min(0.005, nxt - now)); continue
            nxt = now + 1.0 / args.hz
            f = latest["f"]
            if f is None:
                continue
            try:
                link.send_frame(f)
            except (OSError, ConnectionError, json.JSONDecodeError) as e:
                print(f"  [send failed] {type(e).__name__}: {e}", flush=True)
                link.fail(e); time.sleep(1.0)

    threading.Thread(target=worker, daemon=True).start()
    print("press q or Esc to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            latest["f"] = frame.copy()
            people, ms, rtt, sent, err = link.snapshot()
            cv2.imshow("rehab score", draw(frame, people, ms, rtt, sent, err, args.hz))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
