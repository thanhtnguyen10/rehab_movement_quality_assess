import torch
import torch.nn as nn

EDGES = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(1,7),(2,8),(7,8),
         (7,9),(8,10),(9,11),(10,12)]

EDGES_H36M = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
              (8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]

def adjacency(n=13):
    a = torch.eye(n)
    for i, j in (EDGES_H36M if n == 17 else EDGES):
        if i < n and j < n:
            a[i, j] = a[j, i] = 1.0
    d = a.sum(-1).pow(-0.5)
    return d[:, None] * a * d[None, :]

class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(cout),
        )
        self.res = (nn.Identity() if cin == cout and stride == 1
                    else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, A):

        r = self.res(x)
        x = torch.einsum('bctv,vw->bctw', self.gcn(x), A)
        return self.act(self.tcn(x) + r)

class ScoreNet(nn.Module):
    def __init__(self, in_ch=3, n_joints=13, n_ex=5, hidden=64, dual=False, ori_ch=4):
        super().__init__()
        self.dual = dual
        self.register_buffer('A', adjacency(n_joints))
        self.bn = nn.BatchNorm1d(in_ch * n_joints)
        self.bn_ori = nn.BatchNorm1d(ori_ch * n_joints) if dual else None

        def branch(c):
            return nn.ModuleList([STGCNBlock(c, hidden),
                                  STGCNBlock(hidden, hidden),
                                  STGCNBlock(hidden, hidden * 2, stride=2)])
        self.pos = branch(in_ch)
        self.ori = branch(ori_ch) if dual else None

        feat = hidden * 2 * (2 if dual else 1)
        self.ex_emb = nn.Embedding(n_ex + 1, 16)
        self.head = nn.Sequential(
            nn.Linear(feat * 2 + 16, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def _encode(self, x, blocks, bn=None):
        B, T, V, C = x.shape
        bn = bn or self.bn
        x = bn(x.permute(0, 3, 2, 1).reshape(B, C * V, T)).reshape(B, C, V, T)
        x = x.permute(0, 1, 3, 2)
        for b in blocks:
            x = b(x, self.A)
        return torch.cat([x.mean((2, 3)), x.amax(3).amax(2)], -1)

    def forward(self, pos, ex, ori=None):
        f = self._encode(pos, self.pos)
        if self.dual and ori is not None:
            f = torch.cat([f, self._encode(ori, self.ori, self.bn_ori)], -1)
        return self.head(torch.cat([f, self.ex_emb(ex)], -1)).squeeze(-1)
