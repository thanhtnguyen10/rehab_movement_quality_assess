import os, glob, re
import numpy as np
import pandas as pd

KINECT = ['SpineBase','SpineMid','Neck','Head','ShoulderLeft','ElbowLeft','WristLeft',
          'HandLeft','ShoulderRight','ElbowRight','WristRight','HandRight','HipLeft',
          'KneeLeft','AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight',
          'FootRight','SpineShoulder','HandTipLeft','ThumbLeft','HandTipRight','ThumbRight']

COCO = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',
        'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
        'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

K2C = {'nose':'Head','left_shoulder':'ShoulderLeft','right_shoulder':'ShoulderRight',
       'left_elbow':'ElbowLeft','right_elbow':'ElbowRight','left_wrist':'WristLeft',
       'right_wrist':'WristRight','left_hip':'HipLeft','right_hip':'HipRight',
       'left_knee':'KneeLeft','right_knee':'KneeRight','left_ankle':'AnkleLeft',
       'right_ankle':'AnkleRight'}
USE_JOINTS = [c for c in COCO if c in K2C]
KIDX = [KINECT.index(K2C[c]) for c in USE_JOINTS]

H36M = ['Hip','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','Spine','Thorax',
        'Neck','Head','LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist']
H2K = {'Hip':'SpineBase','RHip':'HipRight','RKnee':'KneeRight','RAnkle':'AnkleRight',
       'LHip':'HipLeft','LKnee':'KneeLeft','LAnkle':'AnkleLeft','Spine':'SpineMid',
       'Thorax':'SpineShoulder','Neck':'Neck','Head':'Head','LShoulder':'ShoulderLeft',
       'LElbow':'ElbowLeft','LWrist':'WristLeft','RShoulder':'ShoulderRight',
       'RElbow':'ElbowRight','RWrist':'WristRight'}
H36M_KIDX = [KINECT.index(H2K[j]) for j in H36M]

def normalize_h36m(seq):
    if len(seq) == 0:
        return seq
    seq = seq - seq[:, 0:1, :]
    scale = np.linalg.norm(seq[:, H36M.index('Thorax')], axis=-1).mean()
    return seq / (scale + 1e-6)

def read_joint_csv(path, n_joints=25, stride=4, keep=3):
    vals = []
    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or not any(ch.isdigit() for ch in line):
                continue
            parts = [p for p in line.split(',') if p.strip() != '']
            if len(parts) < n_joints * stride:
                continue
            try:
                row = np.array([float(p) for p in parts[:n_joints * stride]], np.float32)
            except ValueError:
                continue
            vals.append(row.reshape(n_joints, stride)[:, :keep])
    if not vals:
        return np.zeros((0, n_joints, keep), np.float32)
    return np.stack(vals)

def normalize(seq):
    if len(seq) == 0:
        return seq
    names = USE_JOINTS
    lh, rh = names.index('left_hip'), names.index('right_hip')
    ls, rs = names.index('left_shoulder'), names.index('right_shoulder')
    root = (seq[:, lh] + seq[:, rh]) / 2.0
    seq = seq - root[:, None, :]
    shoulder = (seq[:, ls] + seq[:, rs]) / 2.0
    scale = np.linalg.norm(shoulder, axis=-1).mean()
    return seq / (scale + 1e-6)

def parse_labels(root):
    out = {}
    for f in glob.glob(os.path.join(root, '**', 'ClinicalAssessment_*.xlsx'), recursive=True):
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        for _, row in df.iterrows():
            sid = str(row.get('Subject ID', '')).strip()
            if not sid or sid.lower() == 'nan':
                continue
            for ex in range(1, 6):
                rec = {}
                for fam in ('TS', 'PO', 'CF'):
                    col = f'clinical {fam} Ex#{ex}'
                    if col in df.columns and pd.notna(row[col]):
                        rec[fam] = float(row[col])
                if rec:
                    out[(sid, ex)] = rec
    return out

def build_dataset(root, window=64, stride=32, min_frames=32, layout='coco13'):
    labels = parse_labels(root)
    samples = []
    for pos_path in glob.glob(os.path.join(root, '**', 'JointPosition*.csv'), recursive=True):
        parts = pos_path.split(os.sep)
        try:
            es = [p for p in parts if re.fullmatch(r'Es\d+', p)][-1]
            ex = int(es[2:])
            sid = [p for p in parts if re.fullmatch(r'(E|NE|B|P|S)_ID\d+', p)][-1]
            group = parts[parts.index(sid) - 1]
        except (IndexError, ValueError):
            continue
        if ex > 5 or (sid, ex) not in labels:
            continue

        seq = read_joint_csv(pos_path)
        if len(seq) < min_frames:
            continue
        if layout == 'h36m17':
            seq = normalize_h36m(seq[:, H36M_KIDX, :])
            jidx = H36M_KIDX
        else:
            seq = normalize(seq[:, KIDX, :])
            jidx = KIDX

        ori_path = pos_path.replace('JointPosition', 'JointOrientation')
        ori = read_joint_csv(ori_path, stride=4, keep=4) if os.path.exists(ori_path) else None
        if ori is not None and len(ori) >= len(seq):
            ori = ori[:len(seq)][:, jidx, :]
        else:
            ori = None

        for s in range(0, max(1, len(seq) - window + 1), stride):
            w = seq[s:s + window]
            if len(w) < window:
                w = np.concatenate([w, np.repeat(w[-1:], window - len(w), 0)])
            wo = None
            if ori is not None:
                wo = ori[s:s + window]
                if len(wo) < window:
                    wo = np.concatenate([wo, np.repeat(wo[-1:], window - len(wo), 0)])
            samples.append(dict(pos=w.astype(np.float32),
                                ori=None if wo is None else wo.astype(np.float32),
                                subject=sid, group=group, exercise=ex,
                                **labels[(sid, ex)]))
    return samples
