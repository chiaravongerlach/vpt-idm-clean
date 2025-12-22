#!/usr/bin/env python3
"""
final training ablation pipeline for the results 

run with this command to reproduce what I ran: 

nohup python experiments/mspacman/train_ablation_final.py --epochs 15 --model all --loss weighted --init_priors 1 --seed 42 > ablation_overnight.log 2>&1 &


trains the 9 way and factorized classifier 
Input: a sequence of 16 frames of size 84×84 (from the .npz).
for 9way:
output : NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT

"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
import math, argparse, pickle
from tqdm import tqdm

print("=" * 70)
print("FINAL TRAINING")
print("=" * 70)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

H_WEIGHTS = torch.tensor([1.5, 1.0, 0.7], dtype=torch.float)
V_WEIGHTS = torch.tensor([1.8, 0.6, 1.0], dtype=torch.float)
H_PRIORS = torch.tensor([0.267, 0.321, 0.412])
V_PRIORS = torch.tensor([0.218, 0.496, 0.286])

class PositionalEncoding(nn.Module):
    def __init__(self, d, m=16):
        super().__init__()
        pe = torch.zeros(1, m, d)
        pos = torch.arange(m).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class IDMBackbone(nn.Module):
    def __init__(self, seq=16, img=84, hid=512):
        super().__init__()
        self.seq, self.img, self.hid = seq, img, hid
        self.tc = nn.Conv3d(1, 32, 3, padding=1)
        self.c1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(256, hid, 3, stride=2, padding=1)
        self.bn1, self.bn2 = nn.BatchNorm2d(64), nn.BatchNorm2d(128)
        self.bn3, self.bn4 = nn.BatchNorm2d(256), nn.BatchNorm2d(hid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pos = PositionalEncoding(hid, seq)
        enc = nn.TransformerEncoderLayer(hid, 4, hid*4, 0.1, 'relu', batch_first=True)
        self.trans = nn.TransformerEncoder(enc, 2)
    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.tc(x.unsqueeze(1).float() / 255.0))
        x = x.permute(0,2,1,3,4).reshape(B*self.seq, 32, self.img, self.img)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = F.relu(self.bn4(self.c4(x)))
        x = self.pool(x).view(B*self.seq, self.hid).view(B, self.seq, self.hid)
        return self.trans(self.pos(x))[:, self.seq//2, :]

class IDM9Way(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = IDMBackbone()
        self.head = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128,9))
    def forward(self, x):
        return self.head(self.backbone(x))

class IDMFactorized(nn.Module):
    def __init__(self, init_priors=False):
        super().__init__()
        self.backbone = IDMBackbone()
        self.hhead = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128,3))
        self.vhead = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128,3))
        if init_priors:
            with torch.no_grad():
                self.hhead[-1].bias.copy_(torch.log(H_PRIORS.clamp(min=1e-6)).to(self.hhead[-1].bias.device))
                self.vhead[-1].bias.copy_(torch.log(V_PRIORS.clamp(min=1e-6)).to(self.vhead[-1].bias.device))

    def forward(self, x):
        f = self.backbone(x)
        return self.hhead(f), self.vhead(f)

class DS(Dataset):
    def __init__(self, p):
        d = np.load(p)
        self.seq, self.l9, self.lh, self.lv = d['sequences'], d['labels_9way'], d['labels_h'], d['labels_v']
    def __len__(self): return len(self.seq)
    def __getitem__(self, i):
        return (torch.from_numpy(self.seq[i].copy()), torch.tensor(self.l9[i], dtype=torch.long),
                torch.tensor(self.lh[i], dtype=torch.long), torch.tensor(self.lv[i], dtype=torch.long))

def train_9way(model, tl, vl, dev, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    best, hist = 0, []
    for ep in range(epochs):
        model.train()
        for seq, l9, _, _ in tqdm(tl, desc=f"Ep{ep+1} 9way"):
            seq, l9 = seq.to(dev), l9.to(dev)
            opt.zero_grad(); loss = crit(model(seq), l9); loss.backward(); opt.step()
        sch.step()
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for seq, l9, _, _ in vl:
                preds.extend(model(seq.to(dev)).argmax(1).cpu().numpy())
                labs.extend(l9.numpy())
        preds, labs = np.array(preds), np.array(labs)
        f1 = f1_score(labs, preds, average='macro')
        bal = balanced_accuracy_score(labs, preds)
        print(f"Ep{ep+1}: Acc={(preds==labs).mean():.4f} Bal={bal:.4f} F1={f1:.4f}")
        hist.append({'ep':ep+1, 'acc':(preds==labs).mean(), 'bal':bal, 'f1':f1})
        if f1 > best: best = f1; torch.save(model.state_dict(), 'experiments/mspacman/checkpoints/best_9way.pth')
        if (ep+1) % 5 == 0: torch.save(model.state_dict(), f'experiments/mspacman/checkpoints/9way_ep{ep+1}.pth')
    names = ['NOOP','UP','RIGHT','LEFT','DOWN','UPRIGHT','UPLEFT','DOWNRIGHT','DOWNLEFT']
    print("\\n9-WAY REPORT:"); print(classification_report(labs, preds, target_names=names, digits=4, zero_division=0))
    return {'hist': hist, 'f1': f1, 'bal': bal, 'preds': preds, 'labs': labs}

def train_factorized(model, tl, vl, dev, epochs, lr, loss_type='weighted'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    h_w, v_w = H_WEIGHTS.to(dev), V_WEIGHTS.to(dev)
    if loss_type == 'unweighted': crit_h, crit_v = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    elif loss_type == 'weighted': crit_h, crit_v = nn.CrossEntropyLoss(weight=h_w), nn.CrossEntropyLoss(weight=v_w)
    else: crit_h, crit_v = nn.CrossEntropyLoss(weight=h_w, label_smoothing=0.05), nn.CrossEntropyLoss(weight=v_w, label_smoothing=0.05)
    print(f"  Loss: {loss_type}")
    hv9 = {(1,1):0,(1,0):1,(2,1):2,(0,1):3,(1,2):4,(2,0):5,(0,0):6,(2,2):7,(0,2):8}
    best, hist = 0, []
    for ep in range(epochs):
        model.train()
        for seq, _, lh, lv in tqdm(tl, desc=f"Ep{ep+1} Fact"):
            seq, lh, lv = seq.to(dev), lh.to(dev), lv.to(dev)
            opt.zero_grad(); ho, vo = model(seq); loss = crit_h(ho, lh) + crit_v(vo, lv); loss.backward(); opt.step()
        sch.step()
        model.eval()
        hp, vp, hl, vl_arr, p9, l9 = [], [], [], [], [], []
        with torch.no_grad():
            for seq, labs9, labh, labv in vl:
                ho, vo = model(seq.to(dev))
                hpr, vpr = ho.argmax(1).cpu().numpy(), vo.argmax(1).cpu().numpy()
                hp.extend(hpr); vp.extend(vpr); hl.extend(labh.numpy()); vl_arr.extend(labv.numpy())
                for h,v,l in zip(hpr,vpr,labs9.numpy()): p9.append(hv9.get((int(h),int(v)),0)); l9.append(l)
        hp, vp, hl, vl_arr, p9, l9 = map(np.array, [hp, vp, hl, vl_arr, p9, l9])
        h_acc, v_acc = (hp==hl).mean(), (vp==vl_arr).mean()
        joint = ((hp==hl) & (vp==vl_arr)).mean()
        hr = [(hp[hl==c]==c).mean() if (hl==c).sum()>0 else 0 for c in range(3)]
        vr = [(vp[vl_arr==c]==c).mean() if (vl_arr==c).sum()>0 else 0 for c in range(3)]
        f1, bal = f1_score(l9, p9, average='macro'), balanced_accuracy_score(l9, p9)
        print(f"Ep{ep+1}: H={h_acc:.3f} V={v_acc:.3f} J={joint:.3f} F1={f1:.3f} Bal={bal:.3f}")
        print(f"       H_recall(L/N/R): {hr[0]:.2f}/{hr[1]:.2f}/{hr[2]:.2f}")
        print(f"       V_recall(U/N/D): {vr[0]:.2f}/{vr[1]:.2f}/{vr[2]:.2f}")
        hist.append({'ep':ep+1, 'h_acc':h_acc, 'v_acc':v_acc, 'joint':joint, 'h_recall':hr, 'v_recall':vr, 'f1':f1, 'bal':bal})
        if f1 > best: best = f1; torch.save(model.state_dict(), f'experiments/mspacman/checkpoints/best_fact_{loss_type}.pth'); print("       -> Saved best!")
        if (ep+1) % 5 == 0: torch.save(model.state_dict(), f'experiments/mspacman/checkpoints/fact_{loss_type}_ep{ep+1}.pth'); print(f"       -> Checkpoint ep{ep+1}")
    print("\\nH:"); print(classification_report(hl, hp, target_names=['LEFT','NONE','RIGHT'], digits=4, zero_division=0))
    print("\\nV:"); print(classification_report(vl_arr, vp, target_names=['UP','NONE','DOWN'], digits=4, zero_division=0))
    print("\\n9-WAY:"); print(classification_report(l9, p9, target_names=['NOOP','UP','RIGHT','LEFT','DOWN','UPRIGHT','UPLEFT','DOWNRIGHT','DOWNLEFT'], digits=4, zero_division=0))
    return {'hist': hist, 'f1': f1, 'bal': bal, 'joint': joint, 'h_recall': hr, 'v_recall': vr}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', choices=['9way', 'fact', 'all'], default='all')
    parser.add_argument('--loss', choices=['unweighted', 'weighted', 'weighted_smooth'], default='weighted')
    parser.add_argument('--init_priors', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"\\nDevice: {dev}, Seed: {args.seed}")
    print(f"Config: epochs={args.epochs}, batch={args.batch}, lr={args.lr}, loss={args.loss}, init_priors={args.init_priors}")
    os.makedirs('experiments/mspacman/checkpoints', exist_ok=True)
    trn, val = DS('data/mspacman/processed/train.npz'), DS('data/mspacman/processed/val.npz')
    pin, nw = (dev.type == 'cuda'), 4 if dev.type == 'cuda' else 0
    tl = DataLoader(trn, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=pin)
    vl = DataLoader(val, batch_size=args.batch, num_workers=nw, pin_memory=pin)
    print(f"Train: {len(trn):,}, Val: {len(val):,}")
    results = {}
    if args.model in ['9way', 'all']:
        print("\\n" + "="*70 + "\\nTRAINING 9-WAY\\n" + "="*70)
        set_seed(args.seed); m = IDM9Way().to(dev); print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
        results['9way'] = train_9way(m, tl, vl, dev, args.epochs, args.lr)
    if args.model in ['fact', 'all']:
        print("\\n" + "="*70 + f"\\nTRAINING FACTORIZED ({args.loss})\\n" + "="*70)
        set_seed(args.seed); m = IDMFactorized(init_priors=bool(args.init_priors)).to(dev); print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
        results['factorized'] = train_factorized(m, tl, vl, dev, args.epochs, args.lr, args.loss)
    with open('experiments/mspacman/ablation_results.pkl', 'wb') as f: pickle.dump(results, f)
    print("\\n" + "="*70 + "\\nFINAL COMPARISON\\n" + "="*70)
    if '9way' in results: print(f"9-WAY: F1={results['9way']['f1']:.4f} Bal={results['9way']['bal']:.4f}")
    if 'factorized' in results: r=results['factorized']; print(f"FACT:  F1={r['f1']:.4f} Bal={r['bal']:.4f} J={r['joint']:.3f} H={r['h_recall']} V={r['v_recall']}")
    if '9way' in results and 'factorized' in results: d=results['factorized']['f1']-results['9way']['f1']; print(f"Delta F1: {d:+.4f}")

if __name__ == '__main__': 
    main()