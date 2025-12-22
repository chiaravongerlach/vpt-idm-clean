#!/usr/bin/env python3
"""
some evals and analysis on breakout game behavior from initial training stage 

"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from vpt_breakout_model import VPTBreakoutIDM
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import os

OUTPUT_DIR = "report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
class BreakoutDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.sequences = data['sequences']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.sequences[idx]).float() / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

def plot_dataset_distribution(labels, save_path):
    """Plot the class distribution in the dataset"""
    action_names = ['NOOP', 'FIRE', 'LEFT', 'RIGHT', 'LEFTFIRE']
    counts = Counter(labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    actions = [action_names[i] for i in range(5)]
    values = [counts.get(i, 0) for i in range(5)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    bars = ax1.bar(actions, values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Action', fontsize=12)
    ax1.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    non_zero = [(action_names[i], counts.get(i, 0)) for i in range(5) if counts.get(i, 0) > 0]
    pie_labels, pie_values = zip(*non_zero)
    pie_colors = [colors[action_names.index(l)] for l in pie_labels]
    
    wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                        colors=pie_colors, explode=[0.05]*len(pie_values),
                                        shadow=True, startangle=90)
    ax2.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    """Plot confusion matrix"""
    action_names = ['NOOP', 'FIRE', 'LEFT', 'RIGHT', 'LEFTFIRE']
    
    # Filter to only classes that appear in data
    present_classes = sorted(set(y_true) | set(y_pred))
    present_names = [action_names[i] for i in present_classes]
    
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=present_names,
                yticklabels=present_names, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_title(f'{title} (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', xticklabels=present_names,
                yticklabels=present_names, ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_title(f'{title} (Normalized by True Label)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_prediction_comparison(y_true, y_pred, save_path):
    """Compare prediction distribution vs actual distribution"""
    action_names = ['NOOP', 'FIRE', 'LEFT', 'RIGHT', 'LEFTFIRE']
    
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    x = np.arange(5)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    true_vals = [true_counts.get(i, 0) for i in range(5)]
    pred_vals = [pred_counts.get(i, 0) for i in range(5)]
    
    bars1 = ax.bar(x - width/2, true_vals, width, label='Ground Truth', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_vals, width, label='Model Predictions', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xlabel('Action', fontsize=12)
    ax.set_title('Ground Truth vs Model Predictions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_sample_sequences(dataset, model, device, save_path, num_samples=4):
    """Visualize sample sequences with predictions"""
    action_names = ['NOOP', 'FIRE', 'LEFT', 'RIGHT', 'LEFTFIRE']
    
    fig, axes = plt.subplots(num_samples, 8, figsize=(16, num_samples * 2))
    
    # Get samples with different actions
    indices_by_action = {i: [] for i in range(5)}
    for idx in range(len(dataset)):
        label = dataset.labels[idx]
        if len(indices_by_action[label]) < num_samples:
            indices_by_action[label].append(idx)
    
    # Get one sample from each action type
    sample_indices = []
    for action in [0, 2, 3, 1]:  # NOOP, LEFT, RIGHT, FIRE
        if indices_by_action[action]:
            sample_indices.append(indices_by_action[action][0])
    
    model.eval()
    for row, idx in enumerate(sample_indices[:num_samples]):
        sequence, label = dataset[idx]
        
        with torch.no_grad():
            output = model(sequence.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
        
        # Show frames 0, 2, 4, 6, 8, 10, 12, 14 (every other frame)
        for col, frame_idx in enumerate([0, 2, 4, 6, 8, 10, 12, 14]):
            ax = axes[row, col]
            ax.imshow(sequence[frame_idx].numpy(), cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f'True: {action_names[label]}\nPred: {action_names[pred]}', 
                             fontsize=10, rotation=0, ha='right', va='center')
            if row == 0:
                ax.set_title(f'Frame {frame_idx}', fontsize=9)
    
    plt.suptitle('Sample Sequences: Ground Truth vs Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def generate_report_stats(y_true, y_pred):
    """Generate statistics for the report"""
    action_names = ['NOOP', 'FIRE', 'LEFT', 'RIGHT', 'LEFTFIRE']
    
    print("\n" + "="*70)
    print("REPORT STATISTICS")
    print("="*70)
    
    # Overall metrics
    overall_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
    
    print(f"\nOVERALL METRICS:")
    print(f"   Overall Accuracy:  {overall_acc:.2f}%")
    print(f"   Balanced Accuracy: {balanced_acc:.2f}%")
    print(f"   Random Baseline:   20.00% (5 classes)")
    
    # Per-class metrics
    print(f"\nPER-CLASS BREAKDOWN:")
    print(f"   {'Action':<10} {'Samples':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"   {'-'*40}")
    
    for i in range(5):
        mask = np.array(y_true) == i
        n_samples = mask.sum()
        if n_samples > 0:
            correct = (np.array(y_pred)[mask] == i).sum()
            acc = correct / n_samples * 100
            print(f"   {action_names[i]:<10} {n_samples:>8} {correct:>8} {acc:>9.1f}%")
        else:
            print(f"   {action_names[i]:<10} {0:>8} {'N/A':>8} {'N/A':>10}")
    
    # Prediction distribution
    pred_counts = Counter(y_pred)
    print(f"\nPREDICTION DISTRIBUTION:")
    for i in range(5):
        count = pred_counts.get(i, 0)
        pct = count / len(y_pred) * 100
        print(f"   {action_names[i]:<10}: {count:>6} ({pct:>5.1f}%)")
    
    # Key finding
    print(f"\nKEY FINDING:")
    if pred_counts.get(0, 0) / len(y_pred) > 0.95:
        print("   Model predicts NOOP for >95% of samples")
        print("   This indicates the model is exploiting class imbalance")
        print("  Despite 78% overall accuracy, balanced accuracy is much lower")
    
    return {
        'overall_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'per_class': classification_report(y_true, y_pred, target_names=action_names, output_dict=True)
    }

def main():
    print("="*70)
    print("VPT BREAKOUT IDM - EVALUATION & VISUALIZATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load metadata
    with open('breakout_processed/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BreakoutDataset('breakout_processed/train.npz')
    val_dataset = BreakoutDataset('breakout_processed/val.npz')
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 1. Plot dataset distribution
    print("\nGenerating dataset distribution plot...")
    all_labels = np.concatenate([train_dataset.labels, val_dataset.labels])
    plot_dataset_distribution(all_labels, f'{OUTPUT_DIR}/1_dataset_distribution.png')
    
    # 2. Create and evaluate untrained model (baseline)
    print("\nEvaluating untrained model (random baseline)...")
    model = VPTBreakoutIDM(num_actions=5).to(device)
    model.eval()
    
    untrained_preds = []
    untrained_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            outputs = model(sequences.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            untrained_preds.extend(preds)
            untrained_labels.extend(labels.numpy())
    
    plot_confusion_matrix(untrained_labels, untrained_preds, 
                         f'{OUTPUT_DIR}/2_confusion_matrix_untrained.png',
                         title="Untrained Model")
    
    # 3. Load trained model and evaluate
    print("\nEvaluating trained model...")
    
    # Try to load best model from either checkpoint directory
    checkpoint_paths = [
        'checkpoints_weighted/best_model.pth',
        'checkpoints/best_model.pth',
        'checkpoints_weighted/checkpoint_epoch_5.pth',
        'checkpoints/checkpoint_epoch_5.pth'
    ]
    
    model_loaded = False
    for cp_path in checkpoint_paths:
        if os.path.exists(cp_path):
            print(f"   Loading {cp_path}...")
            checkpoint = torch.load(cp_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_loaded = True
            break
    
    if not model_loaded:
        print("   No checkpoint found, using current model state")
    
    trained_preds = []
    trained_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            outputs = model(sequences.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            trained_preds.extend(preds)
            trained_labels.extend(labels.numpy())
    
    plot_confusion_matrix(trained_labels, trained_preds,
                         f'{OUTPUT_DIR}/3_confusion_matrix_trained.png',
                         title="Trained Model")
    
    # 4. Prediction comparison
    print("\nGenerating prediction comparison plot...")
    plot_prediction_comparison(trained_labels, trained_preds,
                              f'{OUTPUT_DIR}/4_prediction_comparison.png')
    
    # 5. Sample sequences visualization
    print("\nGenerating sample sequence visualization...")
    plot_sample_sequences(val_dataset, model, device,
                         f'{OUTPUT_DIR}/5_sample_sequences.png')
    
    # 6. Generate report statistics
    stats = generate_report_stats(trained_labels, trained_preds)
    
    # Save stats to file
    with open(f'{OUTPUT_DIR}/evaluation_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print(f"\nOutput files in '{OUTPUT_DIR}/':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   - {f}")

if __name__ == "__main__":
    main()
