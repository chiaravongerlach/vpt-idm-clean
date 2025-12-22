# check whether there is big difference with untrained model to compare my model 

import torch
import numpy as np
import pickle
from vpt_breakout_model import VPTBreakoutIDM
from torch.utils.data import Dataset, DataLoader

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

# Load metadata
with open('breakout_processed/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# load val data
print("Loading validation data...")
val_dataset = BreakoutDataset('breakout_processed/val.npz')
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# untrained model
device = torch.device('cuda')
model = VPTBreakoutIDM(num_actions=5).to(device)
model.eval()

# pred on 1000 samples
all_predictions = []
all_labels = []

print("Getting predictions...")
with torch.no_grad():
    for i, (sequences, labels) in enumerate(val_loader):
        sequences = sequences.to(device)
        outputs = model(sequences)
        predictions = outputs.argmax(dim=1).cpu().numpy()
        
        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())
        
        if len(all_predictions) >= 1000:
            break

all_predictions = np.array(all_predictions[:1000])
all_labels = np.array(all_labels[:1000])
print("prediciton on 1000samples)")
print("\nFirst 30 predictions vs actual:")
print("Predicted:", all_predictions[:30])
print("Actual:   ", all_labels[:30])

pred_unique, pred_counts = np.unique(all_predictions, return_counts=True)
label_unique, label_counts = np.unique(all_labels, return_counts=True)

print("\nPrediction distribution:")
for action, count in zip(pred_unique, pred_counts):
    pct = (count / len(all_predictions)) * 100
    print(f"   Action {action}: {count:4d} ({pct:5.1f}%)")

print("\nActual label distribution:")
for action, count in zip(label_unique, label_counts):
    pct = (count / len(all_labels)) * 100
    print(f"   Action {action}: {count:4d} ({pct:5.1f}%)")

# Check if it's just predicting one class
if len(pred_unique) == 1:
    print("\nModel is only predicting ONE class!")
    print(f"   Only predicting: Action {pred_unique[0]}")
else:
    print(f"\nmodel predicts {len(pred_unique)} different classes")
    
# Accuracy
accuracy = (all_predictions == all_labels).mean() * 100
print(f"\naccuracy on these 1000 samples: {accuracy:.2f}%")
