#!/usr/bin/env python3
"""
training for weighted model 
this hopefully adresses the imbalance we saw in the initial run
4 actions only: NOOP, FIRE, LEFT, RIGHT 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from vpt_breakout_model import VPTBreakoutIDM, count_parameters

# Configuration
DATA_DIR = "breakout_processed"
CHECKPOINT_DIR = "checkpoints_weighted"
PLOTS_DIR = "plots_weighted"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 20

NUM_ACTIONS = 4
ACTION_NAMES = ['NOOP', 'FIRE', 'LEFT', 'RIGHT']

def remap_labels(labels):
    remapped = labels.copy()
    remapped[remapped == 4] = 2  # LEFTFIRE -> LEFT
    return remapped

class BreakoutDataset(Dataset):
    """Dataset for Breakout sequences with remapped labels"""
    def __init__(self, npz_path):
        print(f"Loading {npz_path}...")
        data = np.load(npz_path)
        self.sequences = data['sequences']
        # Remap labels: LEFTFIRE -> LEFT
        original_labels = data['labels']
        self.labels = remap_labels(original_labels)
        
        # Print remapping stats
        n_remapped = (original_labels == 4).sum()
        print(f"  Loaded {len(self)} sequences")
        print(f"  Remapped {n_remapped} LEFTFIRE -> LEFT")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.sequences[idx]).float() / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

def get_class_weights(num_classes=4):
    """
    Manual class weights prioritizing LEFT and RIGHT
    0=NOOP, 1=FIRE, 2=LEFT, 3=RIGHT
    """
    weights = np.array([
        0.5,   # NOOP - low weight
        0.5,   # FIRE - low weight  
        2.0,   # LEFT - high weight (includes LEFTFIRE)
        2.0,   # RIGHT - high weight
    ])
    
    # Normalize so mean = 1
    weights = weights / weights.mean()
    return weights

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

def print_per_class_accuracy(preds, labels, num_classes=4):
    """Print accuracy for each class"""
    preds = np.array(preds)
    labels = np.array(labels)
    
    print("\n   Per-class accuracy:")
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).mean() * 100
            print(f"      {ACTION_NAMES[i]:8s}: {class_acc:5.1f}% ({mask.sum():5d} samples)")
        else:
            print(f"      {ACTION_NAMES[i]:8s}: N/A (0 samples)")
    
    # Print prediction distribution
    pred_counts = Counter(preds)
    print("\n   Prediction distribution:")
    for i in range(num_classes):
        count = pred_counts.get(i, 0)
        pct = (count / len(preds)) * 100
        print(f"      {ACTION_NAMES[i]:8s}: {pct:5.1f}% ({count:5d} predicted)")

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load metadata
    with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    # Override num_actions to 4
    metadata['num_actions'] = NUM_ACTIONS
    
    print(f"\nDataset :")
    print(f"   Actions: {NUM_ACTIONS} (LEFTFIRE merged into LEFT)")
    print(f"   Train samples: {metadata['n_train']:,}")
    print(f"   Val samples: {metadata['n_val']:,}")
    
    # Load datasets 
    train_dataset = BreakoutDataset(os.path.join(DATA_DIR, 'train.npz'))
    val_dataset = BreakoutDataset(os.path.join(DATA_DIR, 'val.npz'))
    
    # Get manual class weights
    print(f"\n⚖️  Class weights (manual, prioritizing movement):")
    class_weights = get_class_weights(num_classes=NUM_ACTIONS)
    for i, (name, weight) in enumerate(zip(ACTION_NAMES, class_weights)):
        print(f"      {name:8s}: {weight:.2f}")
    
    # Create weighted loss
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    print(f"\nUsing weighted CrossEntropyLoss")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model with 4 actions
    model = VPTBreakoutIDM(
        num_actions=NUM_ACTIONS,
        sequence_length=metadata['sequence_length'],
        img_size=metadata['img_size']
    ).to(device)
    
    print(f"\nModel: {count_parameters(model):,} parameters")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Print per-class accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print_per_class_accuracy(val_preds, val_labels)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        # this isnt the best model since they are imbalanced so we shouldnt use this as metric 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'metadata': metadata
            }, best_model_path)
            print(f"   Val Acc: {val_acc:.2f}%")
        
        # Plot training curves
        plot_path = os.path.join(PLOTS_DIR, 'training_curves.png')
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    # Final eval
    
    _, final_acc, final_preds, final_labels = validate(model, val_loader, criterion, device)
    print(f"\nFinal Validation Accuracy: {final_acc:.2f}%")
    print_per_class_accuracy(final_preds, final_labels)
    
    print(f"\nResults:")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"\nSaved to:")
    print(f"   Checkpoints: {CHECKPOINT_DIR}/")
    print(f"   Plots: {PLOTS_DIR}/")

if __name__ == "__main__":
    main()