import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, '..', 'models', 'MotionAGFormer')

H36M = ['Hip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','Spine','Thorax',
        'Neck','Head','LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist']

COCO = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',
        'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
        'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
_C = {n: i for i, n in enumerate(COCO)}

def coco_to_h36m(kp):
    T = kp.shape[0]
    out = np.zeros((T, 17, 2), np.float32)
    lhip, rhip = kp[:, _C['left_hip']], kp[:, _C['right_hip']]
    lsho, rsho = kp[:, _C['left_shoulder']], kp[:, _C['right_shoulder']]
    hip = (lhip + rhip) / 2
    thorax = (lsho + rsho) / 2
    out[:, 0] = hip
    out[:, 1], out[:, 2], out[:, 3] = rhip, kp[:, _C['right_knee']], kp[:, _C['right_ankle']]
    out[:, 4], out[:, 5], out[:, 6] = lhip, kp[:, _C['left_knee']], kp[:, _C['left_ankle']]
    out[:, 7] = (hip + thorax) / 2
    out[:, 8] = thorax
    out[:, 9] = (thorax + kp[:, _C['nose']]) / 2
    out[:, 10] = kp[:, _C['nose']]
    out[:, 11], out[:, 12], out[:, 13] = lsho, kp[:, _C['left_elbow']], kp[:, _C['left_wrist']]
    out[:, 14], out[:, 15], out[:, 16] = rsho, kp[:, _C['right_elbow']], kp[:, _C['right_wrist']]
    return out

def normalize_screen(kp, w, h):
    return (kp / w * 2 - np.array([1, h / w], np.float32)).astype(np.float32)

class Lifter:
    def __init__(self, ckpt=None, n_frames=81, device='cuda'):
        repo = os.path.abspath(_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from model.MotionAGFormer import MotionAGFormer
        self.n_frames, self.device = n_frames, device
        self.net = MotionAGFormer(n_layers=26, dim_in=3, dim_feat=64, dim_rep=512,
                                  dim_out=3, n_frames=n_frames, neighbour_num=2)
        ckpt = ckpt or os.path.join(_HERE, '..', 'models', 'motionagformer-s-h36m.pth.tr')
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
        self.net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
        self.net.to(device).eval()

    @torch.no_grad()
    def lift(self, kp2d, conf, w, h):
        x = coco_to_h36m(kp2d)
        x = normalize_screen(x, w, h)
        c = np.zeros((x.shape[0], 17, 1), np.float32)
        c[:, [1, 2, 3]] = conf[:, [_C['right_hip'], _C['right_knee'], _C['right_ankle']]][..., None]
        c[:, [4, 5, 6]] = conf[:, [_C['left_hip'], _C['left_knee'], _C['left_ankle']]][..., None]
        c[:, [11, 12, 13]] = conf[:, [_C['left_shoulder'], _C['left_elbow'], _C['left_wrist']]][..., None]
        c[:, [14, 15, 16]] = conf[:, [_C['right_shoulder'], _C['right_elbow'], _C['right_wrist']]][..., None]
        c[:, [0, 7, 8, 9, 10]] = conf.mean(1)[:, None, None]
        inp = np.concatenate([x, c], -1)

        T = inp.shape[0]
        if T < self.n_frames:
            inp = np.concatenate([inp, np.repeat(inp[-1:], self.n_frames - T, 0)])
        elif T > self.n_frames:
            inp = inp[-self.n_frames:]
        out = self.net(torch.from_numpy(inp)[None].to(self.device))[0].cpu().numpy()
        return out[:T] if T <= self.n_frames else out
