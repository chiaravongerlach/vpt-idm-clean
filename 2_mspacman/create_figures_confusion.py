#!/usr/bin/env python3
"""
final figures found in results 3 wiht confusion matrices 
"""
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Save to results2 to avoid overwriting
OUTPUT_DIR = 'results3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# ============================================
# Model Architecture 
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d, m=16):
        super().__init__()
        pe = torch.zeros(1, m, d)
        pos = torch.arange(m).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x): 
        return x + self.pe[:, :x.size(1)]

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
    def __init__(self):
        super().__init__()
        self.backbone = IDMBackbone()
        self.hhead = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128,3))
        self.vhead = nn.Sequential(nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128,3))
    def forward(self, x):
        f = self.backbone(x)
        return self.hhead(f), self.vhead(f)


# parse from logs
def parse_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Parse 9-way epoch results
    nine_way = {'epochs': [], 'acc': [], 'bal': [], 'f1': []}
    for match in re.finditer(r'Ep(\d+): Acc=([\d.]+) Bal=([\d.]+) F1=([\d.]+)', content):
        nine_way['epochs'].append(int(match.group(1)))
        nine_way['acc'].append(float(match.group(2)))
        nine_way['bal'].append(float(match.group(3)))
        nine_way['f1'].append(float(match.group(4)))
    
    # Parse factorized epoch results
    fact = {'epochs': [], 'h_acc': [], 'v_acc': [], 'joint': [], 'f1': [], 'bal': []}
    for match in re.finditer(r'Ep(\d+): H=([\d.]+) V=([\d.]+) J=([\d.]+) F1=([\d.]+) Bal=([\d.]+)', content):
        fact['epochs'].append(int(match.group(1)))
        fact['h_acc'].append(float(match.group(2)))
        fact['v_acc'].append(float(match.group(3)))
        fact['joint'].append(float(match.group(4)))
        fact['f1'].append(float(match.group(5)))
        fact['bal'].append(float(match.group(6)))
    
    # Parse ALL per-epoch H_recall values
    h_recall_all = []
    for match in re.finditer(r'H_recall\(L/N/R\): ([\d.]+)/([\d.]+)/([\d.]+)', content):
        h_recall_all.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if h_recall_all:
        fact['h_recall'] = h_recall_all[-1]
        fact['h_recall_all'] = h_recall_all
    
    # Parse ALL per-epoch V_recall values
    v_recall_all = []
    for match in re.finditer(r'V_recall\(U/N/D\): ([\d.]+)/([\d.]+)/([\d.]+)', content):
        v_recall_all.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    if v_recall_all:
        fact['v_recall'] = v_recall_all[-1]
        fact['v_recall_all'] = v_recall_all
    
    # Parse 9-way classification report
    nine_way_actions = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
    nine_way_report = {}
    
    for action in nine_way_actions:
        pattern = rf'\s+{action}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
        match = re.search(pattern, content)
        if match:
            nine_way_report[action] = {
                'precision': float(match.group(1)),
                'recall': float(match.group(2)),
                'f1': float(match.group(3)),
                'support': int(match.group(4))
            }
    
    # Parse H classification report
    h_report = {}
    h_actions = ['LEFT', 'NONE', 'RIGHT']
    h_section_match = re.search(r'(?:\\n)?H:\s*(?:\\n)?\s*precision.*?(?=(?:\\n)?V:|$)', content, re.DOTALL)
    if h_section_match:
        h_section = h_section_match.group(0)
        for action in h_actions:
            pattern = rf'\s+{action}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
            match = re.search(pattern, h_section)
            if match:
                h_report[action] = {
                    'precision': float(match.group(1)),
                    'recall': float(match.group(2)),
                    'f1': float(match.group(3)),
                    'support': int(match.group(4))
                }
    
    # Parse V classification report
    v_report = {}
    v_actions = ['UP', 'NONE', 'DOWN']
    v_section_match = re.search(r'(?:\\n)?V:\s*(?:\\n)?\s*precision.*?(?=(?:\\n)?9-WAY:|$)', content, re.DOTALL)
    if v_section_match:
        v_section = v_section_match.group(0)
        for action in v_actions:
            pattern = rf'\s+{action}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
            match = re.search(pattern, v_section)
            if match:
                v_report[action] = {
                    'precision': float(match.group(1)),
                    'recall': float(match.group(2)),
                    'f1': float(match.group(3)),
                    'support': int(match.group(4))
                }
    
    # Store reports
    nine_way['report'] = nine_way_report
    fact['h_report'] = h_report
    fact['v_report'] = v_report
    
    # Calculate distribution from support values
    if nine_way_report:
        total = sum(r['support'] for r in nine_way_report.values())
        if total > 0:
            nine_way['distribution'] = {k: v['support']/total*100 for k, v in nine_way_report.items()}
            nine_way['total_samples'] = total
    
    if h_report:
        total_h = sum(r['support'] for r in h_report.values())
        if total_h > 0:
            fact['h_distribution'] = {k: v['support']/total_h*100 for k, v in h_report.items()}
    
    if v_report:
        total_v = sum(r['support'] for r in v_report.values())
        if total_v > 0:
            fact['v_distribution'] = {k: v['support']/total_v*100 for k, v in v_report.items()}
    
    return nine_way, fact


# ============================================
# Run inference to get confusion matrices
# ============================================
def run_inference(model_9way_path, model_fact_path, data_path, device):
    """Load models, run inference, return predictions for confusion matrices"""
    from sklearn.metrics import confusion_matrix
    
    # Load data
    data = np.load(data_path)
    sequences = torch.from_numpy(data['sequences'])
    labels_9way = data['labels_9way']
    labels_h = data['labels_h']
    labels_v = data['labels_v']
    
    print(f"  Loaded {len(sequences)} validation samples")
    
    results = {}
    
    # 9-way model inference
    if os.path.exists(model_9way_path):
        print(f"  Running 9-way inference...")
        model_9way = IDM9Way().to(device)
        model_9way.load_state_dict(torch.load(model_9way_path, map_location=device))
        model_9way.eval()
        
        preds_9way = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size].to(device)
                logits = model_9way(batch)
                preds_9way.extend(logits.argmax(1).cpu().numpy())
        
        preds_9way = np.array(preds_9way)
        results['cm_9way'] = confusion_matrix(labels_9way, preds_9way, labels=range(9))
        print(f"    9-way accuracy: {(preds_9way == labels_9way).mean():.4f}")
    
    # Factorized model inference
    if os.path.exists(model_fact_path):
        print(f"  Running factorized inference...")
        model_fact = IDMFactorized().to(device)
        model_fact.load_state_dict(torch.load(model_fact_path, map_location=device))
        model_fact.eval()
        
        preds_h, preds_v = [], []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size].to(device)
                h_logits, v_logits = model_fact(batch)
                preds_h.extend(h_logits.argmax(1).cpu().numpy())
                preds_v.extend(v_logits.argmax(1).cpu().numpy())
        
        preds_h = np.array(preds_h)
        preds_v = np.array(preds_v)
        
        results['cm_h'] = confusion_matrix(labels_h, preds_h, labels=range(3))
        results['cm_v'] = confusion_matrix(labels_v, preds_v, labels=range(3))
        
        # Reconstruct 9-way predictions
        HV_TO_9 = {
            (1,1): 0, (1,0): 1, (2,1): 2, (0,1): 3, (1,2): 4,
            (2,0): 5, (0,0): 6, (2,2): 7, (0,2): 8
        }
        preds_9way_recon = np.array([HV_TO_9[(h, v)] for h, v in zip(preds_h, preds_v)])
        results['cm_fact_9way'] = confusion_matrix(labels_9way, preds_9way_recon, labels=range(9))
        
        print(f"    H accuracy: {(preds_h == labels_h).mean():.4f}")
        print(f"    V accuracy: {(preds_v == labels_v).mean():.4f}")
        print(f"    Joint accuracy: {((preds_h == labels_h) & (preds_v == labels_v)).mean():.4f}")
    
    return results


def plot_confusion_matrix(cm, labels, title, ax, normalize=True):
    """Plot a single confusion matrix"""
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    else:
        cm_norm = cm
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    # Add text annotations
    thresh = 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    
    return im

# main script 

log_path = 'ablation_overnight.log'
if not os.path.exists(log_path):
    print(f"ERROR: {log_path} not found!")
    exit(1)

nine_way, fact = parse_log(log_path)

if not nine_way['epochs'] or not fact['epochs']:
    print("ERROR: Could not parse results from log")
    exit(1)

print(f"\nParsed from log:")
print(f"  9-way epochs: {len(nine_way['epochs'])}")
print(f"  Factorized epochs: {len(fact['epochs'])}")
print(f"  9-way report classes: {list(nine_way.get('report', {}).keys())}")
print(f"  H report classes: {list(fact.get('h_report', {}).keys())}")
print(f"  V report classes: {list(fact.get('v_report', {}).keys())}")
print(f"  Total validation samples: {nine_way.get('total_samples', 'N/A')}")

# Action names
ACTIONS_9WAY = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
ACTIONS_H = ['LEFT', 'NONE', 'RIGHT']
ACTIONS_V = ['UP', 'NONE', 'DOWN']

# Print parsed values
if 'distribution' in nine_way:
    print(f"\n9-way distribution (from log):")
    for name in ACTIONS_9WAY:
        pct = nine_way['distribution'].get(name, 0)
        support = nine_way['report'].get(name, {}).get('support', 0)
        print(f"  {name}: {pct:.2f}% (n={support})")

if 'h_distribution' in fact and 'v_distribution' in fact:
    print(f"\nFactorized distribution (from log):")
    h_dist = fact['h_distribution']
    v_dist = fact['v_distribution']
    print(f"  H: LEFT={h_dist.get('LEFT',0):.1f}%, NONE={h_dist.get('NONE',0):.1f}%, RIGHT={h_dist.get('RIGHT',0):.1f}%")
    print(f"  V: UP={v_dist.get('UP',0):.1f}%, NONE={v_dist.get('NONE',0):.1f}%, DOWN={v_dist.get('DOWN',0):.1f}%")

# Validate critical data
if 'distribution' not in nine_way or not nine_way['distribution']:
    print("\nERROR: Could not parse 9-way distribution from log")
    exit(1)

if 'h_recall' not in fact or 'v_recall' not in fact:
    print("\nERROR: Could not parse h_recall/v_recall from log")
    exit(1)

h_recall = fact['h_recall']
v_recall = fact['v_recall']


# ============================================
# Run inference for confusion matrices
# ============================================
print(f"\n" + "="*70)
print("RUNNING INFERENCE FOR CONFUSION MATRICES")
print("="*70)

# Find checkpoints and data
checkpoint_dir = 'checkpoints'
data_paths = [
    'data/mspacman/processed/val.npz',
    '../data/mspacman/processed/val.npz',
    os.path.expanduser('~/vpt-idm/data/mspacman/processed/val.npz'),
]

model_9way_path = os.path.join(checkpoint_dir, 'best_9way.pth')
model_fact_path = os.path.join(checkpoint_dir, 'best_fact_weighted.pth')

# Find validation data
val_path = None
for path in data_paths:
    if os.path.exists(path):
        val_path = path
        break

confusion_results = {}
if val_path and os.path.exists(model_9way_path) and os.path.exists(model_fact_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print(f"  Data: {val_path}")
    print(f"  9-way checkpoint: {model_9way_path}")
    print(f"  Factorized checkpoint: {model_fact_path}")
    
    confusion_results = run_inference(model_9way_path, model_fact_path, val_path, device)
else:
    print("  WARNING: Could not find checkpoints or data for confusion matrices")
    print(f"    9-way checkpoint exists: {os.path.exists(model_9way_path)}")
    print(f"    Factorized checkpoint exists: {os.path.exists(model_fact_path)}")
    print(f"    Validation data found: {val_path is not None}")


# ============================================
# Generate Figures
# ============================================
print(f"\n" + "="*70)
print(f"GENERATING FIGURES (saving to {OUTPUT_DIR}/)")
print("="*70)


# ============================================
# Figure 1: Class Distribution
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts_9way = [nine_way['distribution'].get(name, 0) for name in ACTIONS_9WAY]
colors = ['red' if c < 2 else 'steelblue' for c in counts_9way]

bars = axes[0].bar(ACTIONS_9WAY, counts_9way, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Percentage (%)')
axes[0].set_xlabel('Action')
axes[0].set_title('9-Way Action Distribution\n(Red = classes < 2%)')
axes[0].axhline(y=2, color='red', linestyle='--', linewidth=2, label='2% threshold')
axes[0].axhline(y=100/9, color='gray', linestyle=':', alpha=0.7, label=f'Balanced ({100/9:.1f}%)')
axes[0].legend(loc='upper right')
axes[0].tick_params(axis='x', rotation=45)

x = np.arange(3)
width = 0.35
h_counts = [fact['h_distribution'].get(name, 0) for name in ACTIONS_H]
v_counts = [fact['v_distribution'].get(name, 0) for name in ACTIONS_V]

bars1 = axes[1].bar(x - width/2, h_counts, width, label='Horizontal', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = axes[1].bar(x + width/2, v_counts, width, label='Vertical', color='coral', alpha=0.8, edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['LEFT/UP', 'NONE', 'RIGHT/DOWN'])
axes[1].set_ylabel('Percentage (%)')
axes[1].set_title('Factorized Distribution\n(All classes well-represented)')
axes[1].axhline(y=100/3, color='gray', linestyle=':', alpha=0.7, label=f'Balanced ({100/3:.1f}%)')
axes[1].legend()
axes[1].set_ylim(0, 55)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig1_class_distribution.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig1_class_distribution.png/pdf")


# ============================================
# Figure 2: Learning Curves
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

epochs_9 = nine_way['epochs']
epochs_f = fact['epochs']

axes[0,0].plot(epochs_9, nine_way['f1'], 'b-o', label='9-Way', linewidth=2, markersize=5)
axes[0,0].plot(epochs_f, fact['f1'], 'r-s', label='Factorized', linewidth=2, markersize=5)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Macro-F1')
axes[0,0].set_title('(a) Macro-F1')
axes[0,0].legend(loc='lower right')
axes[0,0].set_ylim(0.2, 0.7)
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(epochs_9, nine_way['bal'], 'b-o', label='9-Way', linewidth=2, markersize=5)
axes[0,1].plot(epochs_f, fact['bal'], 'r-s', label='Factorized', linewidth=2, markersize=5)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Balanced Accuracy')
axes[0,1].set_title('(b) Balanced Accuracy')
axes[0,1].legend(loc='lower right')
axes[0,1].set_ylim(0.2, 0.7)
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(epochs_f, fact['h_acc'], 'g-^', label='Horizontal Head', linewidth=2, markersize=5)
axes[1,0].plot(epochs_f, fact['v_acc'], 'm-v', label='Vertical Head', linewidth=2, markersize=5)
axes[1,0].plot(epochs_f, fact['joint'], 'k-d', label='Joint (H ∧ V)', linewidth=2, markersize=5)
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Accuracy')
axes[1,0].set_title('(c) Factorized Per-Head Accuracy')
axes[1,0].legend(loc='lower right')
axes[1,0].set_ylim(0.4, 1.0)
axes[1,0].grid(True, alpha=0.3)

metrics = ['Macro-F1', 'Balanced\nAccuracy']
nine_way_final = [nine_way['f1'][-1], nine_way['bal'][-1]]
fact_final = [fact['f1'][-1], fact['bal'][-1]]

x = np.arange(len(metrics))
width = 0.35
bars1 = axes[1,1].bar(x - width/2, nine_way_final, width, label='9-Way', color='steelblue', alpha=0.8)
bars2 = axes[1,1].bar(x + width/2, fact_final, width, label='Factorized', color='coral', alpha=0.8)

axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(metrics)
axes[1,1].set_ylabel('Score')
axes[1,1].set_title(f'(d) Final Results @ Epoch {epochs_9[-1]}')
axes[1,1].legend()
axes[1,1].set_ylim(0, 0.75)

for bar, val in zip(bars1, nine_way_final):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, fact_final):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_learning_curves.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig2_learning_curves.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig2_learning_curves.png/pdf")


# ============================================
# Figure 3: Per-Axis Performance
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

print(f"\n  Parsed H_recall (L/N/R): {h_recall}")
print(f"  Parsed V_recall (U/N/D): {v_recall}")

bars_h = axes[0].bar(ACTIONS_H, [r*100 for r in h_recall], color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Recall (%)')
axes[0].set_title(f'Horizontal Head\n(Avg: {np.mean(h_recall)*100:.1f}%)')
axes[0].set_ylim(0, 100)
axes[0].axhline(y=90, color='green', linestyle='--', alpha=0.7)
for bar, val in zip(bars_h, h_recall):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

bars_v = axes[1].bar(ACTIONS_V, [r*100 for r in v_recall], color='coral', alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Recall (%)')
axes[1].set_title(f'Vertical Head\n(Avg: {np.mean(v_recall)*100:.1f}%)')
axes[1].set_ylim(0, 100)
axes[1].axhline(y=90, color='green', linestyle='--', alpha=0.7)
for bar, val in zip(bars_v, v_recall):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

joint_metrics = ['H Accuracy', 'V Accuracy', 'Joint\n(H ∧ V)']
joint_values = [fact['h_acc'][-1]*100, fact['v_acc'][-1]*100, fact['joint'][-1]*100]
colors_j = ['steelblue', 'coral', 'purple']
bars_j = axes[2].bar(joint_metrics, joint_values, color=colors_j, alpha=0.8, edgecolor='black')
axes[2].set_ylabel('Accuracy (%)')
axes[2].set_title('Final Epoch Accuracy')
axes[2].set_ylim(0, 100)
for bar, val in zip(bars_j, joint_values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_per_axis_performance.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig3_per_axis_performance.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig3_per_axis_performance.png/pdf")


# ============================================
# Figure 4: Early vs Late Learning
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

epoch_early_idx = min(4, len(fact['f1']) - 1, len(nine_way['f1']) - 1)
epoch_early = epoch_early_idx + 1

categories = [f'Epoch {epoch_early}', f'Epoch {len(nine_way["epochs"])}']
x = np.arange(len(categories))
width = 0.35

f1_9way_compare = [nine_way['f1'][epoch_early_idx], nine_way['f1'][-1]]
f1_fact_compare = [fact['f1'][epoch_early_idx], fact['f1'][-1]]

bars1 = ax.bar(x - width/2, f1_9way_compare, width, label='9-Way', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_fact_compare, width, label='Factorized', color='coral', alpha=0.8)

ax.set_ylabel('Macro-F1')
ax.set_title('Early vs Late Training: Which Model Learns Faster?')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 0.7)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11)

delta_early = f1_fact_compare[0] - f1_9way_compare[0]
delta_late = f1_9way_compare[1] - f1_9way_compare[0]
ax.annotate(f'Factorized leads\nby {delta_early:.3f}', 
            xy=(0.175, f1_fact_compare[0]), xytext=(0.5, 0.45),
            arrowprops=dict(arrowstyle='->', color='coral', lw=2), 
            fontsize=10, color='coral', fontweight='bold')
ax.annotate(f'9-Way catches up\n(+{delta_late:.3f} from ep{epoch_early})', 
            xy=(0.825, f1_9way_compare[1]), xytext=(0.3, 0.60),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=2),
            fontsize=10, color='steelblue', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_early_vs_late.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig4_early_vs_late.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig4_early_vs_late.png/pdf")


# ============================================
# Figure 5: Architecture Diagram
# ============================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

boxes = [
    (0.02, 0.35, 0.12, 0.3, 'Input\n16×84×84\nframes', 'lightgray'),
    (0.18, 0.35, 0.12, 0.3, '3D Conv\nTemporal\nFeatures', 'lightblue'),
    (0.34, 0.35, 0.12, 0.3, '2D CNN\nSpatial\nEncoder', 'lightblue'),
    (0.50, 0.35, 0.12, 0.3, 'Transformer\nTemporal\nAggregation', 'lightblue'),
    (0.66, 0.35, 0.10, 0.3, 'Features\n512-dim', 'lightyellow'),
]

for bx, by, bw, bh, text, color in boxes:
    rect = plt.Rectangle((bx, by), bw, bh, fill=True, facecolor=color,
                          edgecolor='black', linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(bx + bw/2, by + bh/2, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

arrow_style = dict(arrowstyle='->', color='black', lw=2)
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    y = boxes[i][1] + boxes[i][3]/2
    ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=arrow_style)

ax.annotate('', xy=(0.80, 0.72), xytext=(0.76, 0.50), arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
ax.annotate('', xy=(0.80, 0.28), xytext=(0.76, 0.50), arrowprops=dict(arrowstyle='->', color='coral', lw=2))

rect_h = plt.Rectangle((0.80, 0.60), 0.17, 0.25, fill=True,
                        facecolor='steelblue', edgecolor='black', linewidth=2, alpha=0.8, zorder=2)
ax.add_patch(rect_h)
ax.text(0.885, 0.725, 'Horizontal Head\nLEFT / NONE / RIGHT', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white', zorder=3)

rect_v = plt.Rectangle((0.80, 0.15), 0.17, 0.25, fill=True,
                        facecolor='coral', edgecolor='black', linewidth=2, alpha=0.8, zorder=2)
ax.add_patch(rect_v)
ax.text(0.885, 0.275, 'Vertical Head\nUP / NONE / DOWN', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white', zorder=3)

ax.text(0.50, 0.92, 'Factorized IDM Architecture', ha='center', va='center',
        fontsize=14, fontweight='bold')
ax.text(0.50, 0.08, 'Shared backbone learns visual features; independent heads predict orthogonal action components',
        ha='center', va='center', fontsize=10, style='italic')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_architecture.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/fig5_architecture.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig5_architecture.png/pdf")


# ============================================
# Figure 6: Confusion Matrices 
# ============================================
if confusion_results:
    # 6a: 9-way confusion matrix
    if 'cm_9way' in confusion_results:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_confusion_matrix(confusion_results['cm_9way'], ACTIONS_9WAY, 
                             '9-Way Model Confusion Matrix (Normalized)', ax)
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig6a_confusion_9way.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUTPUT_DIR}/fig6a_confusion_9way.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved fig6a_confusion_9way.png/pdf")
    
    # 6b: Factorized H and V confusion matrices
    if 'cm_h' in confusion_results and 'cm_v' in confusion_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        plot_confusion_matrix(confusion_results['cm_h'], ACTIONS_H,
                             'Horizontal Head Confusion Matrix', axes[0])
        plot_confusion_matrix(confusion_results['cm_v'], ACTIONS_V,
                             'Vertical Head Confusion Matrix', axes[1])
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig6b_confusion_factorized_heads.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUTPUT_DIR}/fig6b_confusion_factorized_heads.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved fig6b_confusion_factorized_heads.png/pdf")
    
    # 6c: Reconstructed 9-way from factorized
    if 'cm_fact_9way' in confusion_results:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_confusion_matrix(confusion_results['cm_fact_9way'], ACTIONS_9WAY,
                             'Factorized → 9-Way Reconstruction Confusion Matrix', ax)
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig6c_confusion_factorized_9way.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUTPUT_DIR}/fig6c_confusion_factorized_9way.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved fig6c_confusion_factorized_9way.png/pdf")
    
    # 6d: Side-by-side comparison
    if 'cm_9way' in confusion_results and 'cm_fact_9way' in confusion_results:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        plot_confusion_matrix(confusion_results['cm_9way'], ACTIONS_9WAY,
                             '9-Way Baseline', axes[0])
        plot_confusion_matrix(confusion_results['cm_fact_9way'], ACTIONS_9WAY,
                             'Factorized (Reconstructed)', axes[1])
        
        plt.suptitle('Confusion Matrix Comparison: 9-Way vs Factorized', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig6d_confusion_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{OUTPUT_DIR}/fig6d_confusion_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved fig6d_confusion_comparison.png/pdf")

else:
    print("  Skipped confusion matrices (no inference results)")

#print logs
print("\n" + "="*70)
print("FINAL RESULTS (parsed from ablation_overnight.log)")
print("="*70)
print(f"\n{'Model':<25} {'Macro-F1':<12} {'Balanced Acc':<15} {'Acc / Joint':<12}")
print("-"*70)
print(f"{'9-Way Baseline':<25} {nine_way['f1'][-1]:<12.4f} {nine_way['bal'][-1]:<15.4f} {nine_way['acc'][-1]:<12.4f}")
print(f"{'Factorized (weighted)':<25} {fact['f1'][-1]:<12.4f} {fact['bal'][-1]:<15.4f} {fact['joint'][-1]:<12.4f} (joint)")
print("-"*70)
print(f"{'Delta (Fact - 9way)':<25} {fact['f1'][-1]-nine_way['f1'][-1]:<+12.4f} {fact['bal'][-1]-nine_way['bal'][-1]:<+15.4f}")

print(f"\nPer-Axis Recall (Factorized, Final Epoch) - PARSED FROM LOG:")
print(f"  Horizontal: LEFT={h_recall[0]:.1%}, NONE={h_recall[1]:.1%}, RIGHT={h_recall[2]:.1%}")
print(f"  Vertical:   UP={v_recall[0]:.1%}, NONE={v_recall[1]:.1%}, DOWN={v_recall[2]:.1%}")

print(f"\nFactorized Head Accuracies (Final Epoch) - PARSED FROM LOG:")
print(f"  H Accuracy: {fact['h_acc'][-1]:.1%}")
print(f"  V Accuracy: {fact['v_acc'][-1]:.1%}")
print(f"  Joint (H ∧ V): {fact['joint'][-1]:.1%}")

print(f"\n9-Way Per-Class Performance (from log):")
for name in ACTIONS_9WAY:
    if name in nine_way['report']:
        r = nine_way['report'][name]
        print(f"  {name:<10}: Prec={r['precision']:.2f} Rec={r['recall']:.2f} F1={r['f1']:.2f} (n={r['support']})")

