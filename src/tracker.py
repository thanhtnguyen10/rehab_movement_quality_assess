import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_batch(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-6)

class Track:
    _next = 1

    def __init__(self, box, kpt, score, score_vec=None):
        self.id = Track._next
        Track._next += 1
        self.box, self.kpt, self.score = box, kpt, score
        self.score_vec = score_vec
        self.age = 0
        self.hits = 1
        self.history = [kpt]

    def update(self, box, kpt, score, score_vec=None):
        self.box, self.kpt, self.score = box, kpt, score
        if score_vec is not None:
            self.score_vec = score_vec
        self.age = 0
        self.hits += 1
        self.history.append(kpt)
        if len(self.history) > 512:
            self.history.pop(0)

class ByteTrack:
    def __init__(self, high_thr=0.5, low_thr=0.1, iou_thr=0.3, max_age=30, min_hits=3):
        self.high_thr, self.low_thr = high_thr, low_thr
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.tracks: list[Track] = []

    @staticmethod
    def _kpt_box(kpt, scores, thr=0.3):
        v = kpt[scores > thr]
        if len(v) < 2:
            return None
        return np.array([v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()])

    def _assign(self, tracks, boxes, kpts, scores, svecs=None):
        if not tracks or len(boxes) == 0:
            return list(range(len(boxes)))
        cost = 1.0 - iou_batch([t.box for t in tracks], boxes)
        r, c = linear_sum_assignment(cost)
        matched = set()
        for ti, di in zip(r, c):
            if cost[ti, di] <= 1 - self.iou_thr:
                tracks[ti].update(boxes[di], kpts[di], scores[di],
                                  None if svecs is None else svecs[di])
                matched.add(di)
        return [i for i in range(len(boxes)) if i not in matched]

    def update(self, keypoints, kpt_scores):
        boxes, kpts, confs, svecs = [], [], [], []
        for k, s in zip(keypoints, kpt_scores):
            b = self._kpt_box(k, s)
            if b is not None:
                boxes.append(b); kpts.append(k)
                confs.append(float(s.mean())); svecs.append(s)
        boxes, confs = np.array(boxes), np.array(confs)

        for t in self.tracks:
            t.age += 1

        hi = np.where(confs >= self.high_thr)[0] if len(confs) else np.array([], int)
        lo = np.where((confs < self.high_thr) & (confs >= self.low_thr))[0] if len(confs) else np.array([], int)

        left = self._assign(self.tracks,
                            [boxes[i] for i in hi], [kpts[i] for i in hi],
                            [confs[i] for i in hi], [svecs[i] for i in hi])

        stale = [t for t in self.tracks if t.age > 0]
        self._assign(stale,
                     [boxes[i] for i in lo], [kpts[i] for i in lo],
                     [confs[i] for i in lo], [svecs[i] for i in lo])

        for j in left:
            i = hi[j]
            self.tracks.append(Track(boxes[i], kpts[i], confs[i], svecs[i]))

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.age == 0]
