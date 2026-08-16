import argparse, sys, os, json
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimore_data import build_dataset
from scorenet import ScoreNet

class DS(Dataset):
    def __init__(self, samples, target, mu=0.0, sd=1.0, dual=False):
        self.s, self.t, self.mu, self.sd, self.dual = samples, target, mu, sd, dual

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        d = self.s[i]
        ori = d['ori'] if (self.dual and d['ori'] is not None) else np.zeros((*d['pos'].shape[:2], 4), np.float32)
        return (torch.from_numpy(d['pos']), torch.from_numpy(ori.astype(np.float32)),
                torch.tensor(d['exercise']), torch.tensor((d[self.t] - self.mu) / self.sd, dtype=torch.float32))

def run_fold(tr, va, args, dev):
    ys = np.array([d[args.target] for d in tr], np.float32)
    mu, sd = ys.mean(), ys.std() + 1e-6
    dl_tr = DataLoader(DS(tr, args.target, mu, sd, args.dual), batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(DS(va, args.target, mu, sd, args.dual), batch_size=args.bs, num_workers=4)

    m = ScoreNet(dual=args.dual, n_joints=17 if args.layout == 'h36m17' else 13).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * max(1, len(dl_tr)))
    lossf = torch.nn.SmoothL1Loss()

    best = None
    for ep in range(args.epochs):
        m.train()
        for pos, ori, ex, y in dl_tr:
            pos, ori, ex, y = pos.to(dev), ori.to(dev), ex.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(m(pos, ex, ori if args.dual else None), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()

        m.eval(); P, Y = [], []
        with torch.no_grad():
            for pos, ori, ex, y in dl_va:
                p = m(pos.to(dev), ex.to(dev), ori.to(dev) if args.dual else None)
                P.append(p.cpu().numpy() * sd + mu); Y.append(y.numpy() * sd + mu)
        P, Y = np.concatenate(P), np.concatenate(Y)
        mae = float(np.abs(P - Y).mean())

        if best is None or mae < best['mae']:
            best = dict(mae=mae,
                        pearson=float(pearsonr(P, Y)[0]) if len(set(Y)) > 1 else 0.0,
                        spearman=float(spearmanr(P, Y)[0]) if len(set(Y)) > 1 else 0.0,
                        epoch=ep)
    return best, m, (mu, sd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('KIMORE_ROOT', 'data/KiMoRe'),
                    help='KiMoRe root (or set KIMORE_ROOT)')
    ap.add_argument('--target', default='TS', choices=['TS', 'PO', 'CF'])
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--dual', action='store_true', help='use JointOrientation stream too')
    ap.add_argument('--layout', default='coco13', choices=['coco13', 'h36m17'],
                    help="h36m17 matches the MotionAGFormer 3D deploy path")
    ap.add_argument('--out', default='outputs')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('building dataset...', flush=True)
    samples = build_dataset(args.root, args.window, args.stride, layout=args.layout)
    subs = np.array([d['subject'] for d in samples])
    print(f'{len(samples)} windows from {len(set(subs))} subjects', flush=True)

    gkf = GroupKFold(n_splits=args.folds)
    res = []
    for k, (i_tr, i_va) in enumerate(gkf.split(samples, groups=subs)):
        tr = [samples[i] for i in i_tr]; va = [samples[i] for i in i_va]
        assert not (set(subs[i_tr]) & set(subs[i_va])), 'subject leakage!'
        b, m, norm = run_fold(tr, va, args, dev)
        print(f"fold {k}: MAE {b['mae']:.3f}  pearson {b['pearson']:.3f}  spearman {b['spearman']:.3f}", flush=True)
        res.append(b)
        os.makedirs(args.out, exist_ok=True)
        torch.save({'model': m.state_dict(), 'norm': norm, 'args': vars(args)},
                   os.path.join(args.out, f'fold{k}_{args.target}.pt'))

    for key in ('mae', 'pearson', 'spearman'):
        v = [r[key] for r in res]
        print(f'CV {key}: {np.mean(v):.3f} +/- {np.std(v):.3f}')
    json.dump(res, open(os.path.join(args.out, f'cv_{args.target}.json'), 'w'), indent=2)

if __name__ == '__main__':
    main()
