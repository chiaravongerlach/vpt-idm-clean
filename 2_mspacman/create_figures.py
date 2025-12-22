#!/usr/bin/env python3
"""
creat results2 from overnight logs
"""
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Save to results2 to avoid overwriting
OUTPUT_DIR = 'results2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# ============================================
# Parse EVERYTHING from training log
# ============================================
def parse_log(log_path):
    """Extract ALL metrics from training log - no hardcoding"""
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
    # Pattern matches lines like: "        NOOP     0.9497    0.8610    0.9032       899"
    nine_way_actions = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
    nine_way_report = {}
    
    for action in nine_way_actions:
        # Match the action with its metrics - first occurrence is from 9-way report
        pattern = rf'\s+{action}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
        match = re.search(pattern, content)
        if match:
            nine_way_report[action] = {
                'precision': float(match.group(1)),
                'recall': float(match.group(2)),
                'f1': float(match.group(3)),
                'support': int(match.group(4))
            }
    
    # Parse H classification report (LEFT, NONE, RIGHT)
    h_report = {}
    h_actions = ['LEFT', 'NONE', 'RIGHT']
    
    # Find the H section - after "\nH:" or "\\nH:"
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
    
    # Parse V classification report (UP, NONE, DOWN)
    v_report = {}
    v_actions = ['UP', 'NONE', 'DOWN']
    
    # Find the V section - after "\nV:" or "\\nV:"
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
# Load the log
# ============================================
print("="*70)
print("PARSING TRAINING LOG (no hardcoded values)")
print("="*70)

log_path = 'ablation_overnight.log'
if not os.path.exists(log_path):
    print(f"ERROR: {log_path} not found!")
    print("Make sure ablation_overnight.log is in the current directory")
    exit(1)

nine_way, fact = parse_log(log_path)

# Validate we got data
if not nine_way['epochs']:
    print("ERROR: Could not parse 9-way epoch results from log")
    exit(1)
if not fact['epochs']:
    print("ERROR: Could not parse factorized epoch results from log")
    exit(1)

print(f"\nParsed from log:")
print(f"  9-way epochs: {len(nine_way['epochs'])}")
print(f"  Factorized epochs: {len(fact['epochs'])}")
print(f"  9-way report classes: {list(nine_way.get('report', {}).keys())}")
print(f"  H report classes: {list(fact.get('h_report', {}).keys())}")
print(f"  V report classes: {list(fact.get('v_report', {}).keys())}")
print(f"  Total validation samples: {nine_way.get('total_samples', 'N/A')}")

# Action names (order matters for plotting)
ACTIONS_9WAY = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
ACTIONS_H = ['LEFT', 'NONE', 'RIGHT']
ACTIONS_V = ['UP', 'NONE', 'DOWN']

# Print parsed distribution
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

# Verify we have the critical data
if 'distribution' not in nine_way or not nine_way['distribution']:
    print("\nERROR: Could not parse 9-way distribution from log")
    print("Check that the log contains the classification report")
    exit(1)

if 'h_recall' not in fact or 'v_recall' not in fact:
    print("\nERROR: Could not parse h_recall/v_recall from log")
    exit(1)


# ============================================
# Figure 1: Class Distribution
# ============================================
print(f"\nGenerating figures (saving to {OUTPUT_DIR}/)...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: 9-way distribution from parsed log
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

# Panel B: Factorized distribution from parsed log
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

# Panel A: Macro-F1
axes[0,0].plot(epochs_9, nine_way['f1'], 'b-o', label='9-Way', linewidth=2, markersize=5)
axes[0,0].plot(epochs_f, fact['f1'], 'r-s', label='Factorized', linewidth=2, markersize=5)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Macro-F1')
axes[0,0].set_title('(a) Macro-F1')
axes[0,0].legend(loc='lower right')
axes[0,0].set_ylim(0.2, 0.7)
axes[0,0].grid(True, alpha=0.3)

# Panel B: Balanced Accuracy
axes[0,1].plot(epochs_9, nine_way['bal'], 'b-o', label='9-Way', linewidth=2, markersize=5)
axes[0,1].plot(epochs_f, fact['bal'], 'r-s', label='Factorized', linewidth=2, markersize=5)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Balanced Accuracy')
axes[0,1].set_title('(b) Balanced Accuracy')
axes[0,1].legend(loc='lower right')
axes[0,1].set_ylim(0.2, 0.7)
axes[0,1].grid(True, alpha=0.3)

# Panel C: Factorized Head Accuracies
axes[1,0].plot(epochs_f, fact['h_acc'], 'g-^', label='Horizontal Head', linewidth=2, markersize=5)
axes[1,0].plot(epochs_f, fact['v_acc'], 'm-v', label='Vertical Head', linewidth=2, markersize=5)
axes[1,0].plot(epochs_f, fact['joint'], 'k-d', label='Joint (H ∧ V)', linewidth=2, markersize=5)
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Accuracy')
axes[1,0].set_title('(c) Factorized Per-Head Accuracy')
axes[1,0].legend(loc='lower right')
axes[1,0].set_ylim(0.4, 1.0)
axes[1,0].grid(True, alpha=0.3)

# Panel D: Final Comparison Bar Chart
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
# Figure 3: Per-Axis Performance (from parsed recall)
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

h_recall = fact['h_recall']
v_recall = fact['v_recall']

print(f"\n  Parsed H_recall (L/N/R): {h_recall}")
print(f"  Parsed V_recall (U/N/D): {v_recall}")

# Panel A: Horizontal Head Recall
bars_h = axes[0].bar(ACTIONS_H, [r*100 for r in h_recall], color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Recall (%)')
axes[0].set_title(f'Horizontal Head\n(Avg: {np.mean(h_recall)*100:.1f}%)')
axes[0].set_ylim(0, 100)
axes[0].axhline(y=90, color='green', linestyle='--', alpha=0.7)
for bar, val in zip(bars_h, h_recall):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel B: Vertical Head Recall
bars_v = axes[1].bar(ACTIONS_V, [r*100 for r in v_recall], color='coral', alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Recall (%)')
axes[1].set_title(f'Vertical Head\n(Avg: {np.mean(v_recall)*100:.1f}%)')
axes[1].set_ylim(0, 100)
axes[1].axhline(y=90, color='green', linestyle='--', alpha=0.7)
for bar, val in zip(bars_v, v_recall):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel C: Joint accuracy breakdown
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
# Print Final Summary (ALL FROM LOG)
# ============================================
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

print("\n" + "="*70)
print(f"All figures saved to {OUTPUT_DIR}/")
print("="*70)