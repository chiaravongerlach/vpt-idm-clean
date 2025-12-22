#!/usr/bin/env python3
"""
training script for entire dataset 
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
from vpt_breakout_model import VPTBreakoutIDM, count_parameters

# configs 
DATA_DIR = "breakout_processed"
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 20
VALIDATION_FIRST = True
VALIDATION_SAMPLES = 5000

class BreakoutDataset(Dataset):
    """Dataset for Breakout sequences"""
    def __init__(self, npz_path):
        print(f"Loading {npz_path}...")
        data = np.load(npz_path)
        self.sequences = data['sequences']
        self.labels = data['labels']
        print(f"  Loaded {len(self)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.sequences[idx]).float() / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label
#does one epoch training 
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
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
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
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
    
    return total_loss / len(dataloader), 100. * correct / total

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
    print(f"Saved plot to {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #chekc gpu 
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load metadata
    with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"\nDataset Info:")
    print(f"   Actions: {metadata['num_actions']}")
    print(f"   Train samples: {metadata['n_train']:,}")
    print(f"   Val samples: {metadata['n_val']:,}")
    
    # load
    train_dataset = BreakoutDataset(os.path.join(DATA_DIR, 'train.npz'))
    val_dataset = BreakoutDataset(os.path.join(DATA_DIR, 'val.npz'))
    
    # Quick validation training
    if VALIDATION_FIRST:
        print(f"\n validate on 5k samples first ")
        
        val_indices = np.random.choice(len(train_dataset), VALIDATION_SAMPLES, replace=False)
        val_train_dataset = Subset(train_dataset, val_indices)
        
        val_train_loader = DataLoader(val_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = VPTBreakoutIDM(
            num_actions=metadata['num_actions'],
            sequence_length=metadata['sequence_length'],
            img_size=metadata['img_size']
        ).to(device)
        
        print(f"\nModel: {count_parameters(model):,} parameters")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        print(f"\nTraining for 3 epochs")
        best_val_acc = 0
        
        for epoch in range(3):
            print(f"\nEpoch {epoch + 1}/3:")
            train_loss, train_acc = train_epoch(model, val_train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_val_loader, criterion, device)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            best_val_acc = max(best_val_acc, val_acc)
        
        print(f"\nlidation Training Results:")
        print(f"   Best val accuracy: {best_val_acc:.2f}%")
        
        if best_val_acc < 25:
            print(f"\nWARNING: Accuracy too low (<25%)!")
            return
        elif best_val_acc < 30:
            print(f"\nMarginal results (25-30%).")
        else:
            print(f"\nGood validation results! Proceeding to full training.")
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Full training
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = VPTBreakoutIDM(
        num_actions=metadata['num_actions'],
        sequence_length=metadata['sequence_length'],
        img_size=metadata['img_size']
    ).to(device)
    
    print(f"\nModel: {count_parameters(model):,} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    print(f"\n Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"   Saved checkpoint: {checkpoint_path}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'metadata': metadata
            }, best_model_path)
            print(f"   best model Val Acc: {val_acc:.2f}%")
        
        plot_path = os.path.join(PLOTS_DIR, 'training_curves.png')
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, plot_path)

    print(f"\nfinal Results:")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
