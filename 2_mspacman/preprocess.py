#!/usr/bin/env python3
"""
mspacman preprocessing before training, creating train and val split and factorization labels and 9 way labelse 
creates sequences with the same dimension as breakout 
"""

import os
import tarfile
import numpy as np
from PIL import Image
from collections import Counter
import pickle
from tqdm import tqdm
import re

# config 
MSPACMAN_DIR = "data/mspacman/raw"
OUTPUT_DIR = "data/mspacman/processed"
SEQUENCE_LENGTH = 16
FRAME_SKIP = 4  # Sample every 4th frame same as VPT
IMG_SIZE = 84
TRAIN_SPLIT = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# verify data exists 
print(f"\nVerifying directory structure...")
assert os.path.exists(MSPACMAN_DIR), f"Directory not found: {MSPACMAN_DIR}"
print(f"   {MSPACMAN_DIR} exists")

highscore_dir = os.path.join(MSPACMAN_DIR, 'highscore')
if os.path.exists(highscore_dir):
    print(f"   {highscore_dir} exists")
else:
    print(f"   {highscore_dir} not found (will skip highscore trials)")

# Factorized mapping: action_id -> (horizontal, vertical)
# Horizontal: 0=LEFT, 1=NONE, 2=RIGHT
# Vertical: 0=UP, 1=NONE, 2=DOWN
FACTORIZED_MAP = {
    0: (1, 1), 1: (1, 0), 2: (2, 1), 3: (0, 1), 4: (1, 2),
    5: (2, 0), 6: (0, 0), 7: (2, 2), 8: (0, 2),
    9: (1, 0), 10: (2, 1), 11: (0, 1), 12: (1, 2),
    13: (2, 0), 14: (0, 0), 15: (2, 2), 16: (0, 2), 17: (1, 1),
}

# 9-class mapping (collapse FIRE variants to NOOP)
NINE_CLASS_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 17: 0,
}

# FIRE actions
FIRE_ACTIONS = {9, 10, 11, 12, 13, 14, 15, 16, 17}

print(f"\nConfiguration:")
print(f"   Sequence length: {SEQUENCE_LENGTH} frames")
print(f"   Frame skip: {FRAME_SKIP} (sample every {FRAME_SKIP}th frame)")
print(f"   Effective time span: {SEQUENCE_LENGTH * FRAME_SKIP} raw frames")
print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Train/val split: {TRAIN_SPLIT:.0%}/{1-TRAIN_SPLIT:.0%}")

# Get all trials
all_trials = []
for f in os.listdir(MSPACMAN_DIR):
    if f.endswith('.tar.bz2'):
        all_trials.append(('regular', f.replace('.tar.bz2', '')))

if os.path.exists(highscore_dir):
    for f in os.listdir(highscore_dir):
        if f.endswith('.tar.bz2'):
            all_trials.append(('highscore', f.replace('.tar.bz2', '')))

print(f"\nFound {len(all_trials)} trials:")
print(f"   Regular: {sum(1 for t in all_trials if t[0] == 'regular')}")
print(f"   Highscore: {sum(1 for t in all_trials if t[0] == 'highscore')}")

def extract_frame_number(filename):
    """Extract numeric frame ID for proper sorting"""
    # Handle formats like: "RZ_1234567_1.png" or "1.png" or "frame_001.png"
    basename = os.path.splitext(filename)[0]
    # Find the last number in the filename
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return int(numbers[-1])
    return 0

def load_actions_from_txt(txt_path):
    """Load raw actions from txt file"""
    actions = []
    with open(txt_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 5:
                try:
                    action = int(parts[5])
                    actions.append(action)
                except:
                    pass
    return actions

def load_and_resize_frame(frame_path):
    """Load and resize frame to 84x84 grayscale"""
    img = Image.open(frame_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

# First pass: count total sequences to pre-allocate
print(f"\nCounting sequences...")
total_sequences = 0
trial_infos = []

for trial_type, trial_name in tqdm(all_trials, desc="Counting"):
    try:
        if trial_type == 'highscore':
            tar_path = os.path.join(highscore_dir, f"{trial_name}.tar.bz2")
            txt_path = os.path.join(highscore_dir, f"{trial_name}.txt")
            frames_dir = os.path.join(highscore_dir, trial_name)
        else:
            tar_path = os.path.join(MSPACMAN_DIR, f"{trial_name}.tar.bz2")
            txt_path = os.path.join(MSPACMAN_DIR, f"{trial_name}.txt")
            frames_dir = os.path.join(MSPACMAN_DIR, trial_name)
        
        # Extract if needed
        if not os.path.exists(frames_dir):
            with tarfile.open(tar_path, 'r:bz2') as tar:
                if trial_type == 'highscore':
                    tar.extractall(path=highscore_dir)
                else:
                    tar.extractall(path=MSPACMAN_DIR)
        
        actions = load_actions_from_txt(txt_path)
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        
        # CRITICAL: Sort frames numerically, not lexicographically
        frame_files = sorted(frame_files, key=extract_frame_number)
        
        n_frames = min(len(frame_files), len(actions))
        
        # Calculate sequences with frame skip
        # Need SEQUENCE_LENGTH * FRAME_SKIP frames for one sequence
        frames_needed = (SEQUENCE_LENGTH - 1) * FRAME_SKIP + 1
        n_seq = max(0, (n_frames - frames_needed) // FRAME_SKIP + 1)
        
        if n_seq > 0:
            trial_infos.append({
                'type': trial_type,
                'name': trial_name,
                'frames_dir': frames_dir,
                'txt_path': txt_path,
                'frame_files': frame_files,
                'actions': actions,
                'n_frames': n_frames,
                'n_seq': n_seq,
            })
            total_sequences += n_seq
    except Exception as e:
        print(f"\nError counting {trial_name}: {e}")

print(f"\nTotal sequences to create: {total_sequences:,}")

# Shuffle trials for random split
np.random.shuffle(trial_infos)

# Pre-allocate arrays (memory efficient)
print(f"\nPre-allocating arrays...")
print(f"   Estimated memory: {total_sequences * SEQUENCE_LENGTH * IMG_SIZE * IMG_SIZE / 1e9:.2f} GB")

sequences = np.zeros((total_sequences, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
labels_9way = np.zeros(total_sequences, dtype=np.int64)
labels_h = np.zeros(total_sequences, dtype=np.int64)
labels_v = np.zeros(total_sequences, dtype=np.int64)
labels_fire = np.zeros(total_sequences, dtype=np.int64)
labels_raw = np.zeros(total_sequences, dtype=np.int64)

# Process and fill arrays
print(f"\nProcessing frames...")
idx = 0
raw_action_counts = Counter()

for info in tqdm(trial_infos, desc="Processing"):
    try:
        frame_files = info['frame_files']
        actions = info['actions']
        frames_dir = info['frames_dir']
        n_frames = info['n_frames']
        
        # Create sequences with VPT-style frame skipping
        frames_needed = (SEQUENCE_LENGTH - 1) * FRAME_SKIP + 1
        
        for start in range(0, n_frames - frames_needed + 1, FRAME_SKIP):
            # Sample every FRAME_SKIP-th frame
            seq_frames = []
            for j in range(SEQUENCE_LENGTH):
                frame_idx = start + j * FRAME_SKIP
                frame_path = os.path.join(frames_dir, frame_files[frame_idx])
                seq_frames.append(load_and_resize_frame(frame_path))
            
            # Get action at middle of sampled sequence
            mid_frame_idx = start + (SEQUENCE_LENGTH // 2) * FRAME_SKIP
            raw_action = actions[mid_frame_idx]
            raw_action_counts[raw_action] += 1
            
            # Convert to labels
            label_9way = NINE_CLASS_MAP.get(raw_action, 0)
            h, v = FACTORIZED_MAP.get(raw_action, (1, 1))
            fire = 1 if raw_action in FIRE_ACTIONS else 0
            
            # Store
            sequences[idx] = np.stack(seq_frames, axis=0)
            labels_9way[idx] = label_9way
            labels_h[idx] = h
            labels_v[idx] = v
            labels_fire[idx] = fire
            labels_raw[idx] = raw_action
            idx += 1
            
    except Exception as e:
        print(f"\nError processing {info['name']}: {e}")

# Trim if we have fewer than expected
if idx < total_sequences:
    print(f"\nTrimming arrays from {total_sequences} to {idx}")
    sequences = sequences[:idx]
    labels_9way = labels_9way[:idx]
    labels_h = labels_h[:idx]
    labels_v = labels_v[:idx]
    labels_fire = labels_fire[:idx]
    labels_raw = labels_raw[:idx]

print(f"\nActual sequences created: {idx:,}")

# Train/val split
print(f"\nSplitting train/val...")
n = len(sequences)
n_train = int(n * TRAIN_SPLIT)
perm = np.random.permutation(n)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

# Save
print(f"\nSaving train.npz ({len(train_idx):,} sequences)...")
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'train.npz'),
    sequences=sequences[train_idx],
    labels_9way=labels_9way[train_idx],
    labels_h=labels_h[train_idx],
    labels_v=labels_v[train_idx],
    labels_fire=labels_fire[train_idx],
    labels_raw=labels_raw[train_idx],
)

print(f"Saving val.npz ({len(val_idx):,} sequences)...")
np.savez_compressed(
    os.path.join(OUTPUT_DIR, 'val.npz'),
    sequences=sequences[val_idx],
    labels_9way=labels_9way[val_idx],
    labels_h=labels_h[val_idx],
    labels_v=labels_v[val_idx],
    labels_fire=labels_fire[val_idx],
    labels_raw=labels_raw[val_idx],
)

# Save metadata
metadata = {
    'sequence_length': SEQUENCE_LENGTH,
    'frame_skip': FRAME_SKIP,
    'img_size': IMG_SIZE,
    'n_train': len(train_idx),
    'n_val': len(val_idx),
    'action_names_9way': ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT'],
    'h_names': ['LEFT', 'NONE', 'RIGHT'],
    'v_names': ['UP', 'NONE', 'DOWN'],
    'factorized_map': FACTORIZED_MAP,
    'nine_class_map': NINE_CLASS_MAP,
    'fire_actions': list(FIRE_ACTIONS),
}
with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

# Print distributions
print("\n" + "=" * 70)
print("LABEL DISTRIBUTIONS")
print("=" * 70)

print("\nRaw action distribution:")
for a in sorted(raw_action_counts.keys()):
    count = raw_action_counts[a]
    pct = count / idx * 100
    print(f"   Action {a:2d}: {count:>7,} ({pct:>5.1f}%)")

print("\n9-way distribution:")
names_9 = metadata['action_names_9way']
for c in range(9):
    count = (labels_9way == c).sum()
    pct = count / idx * 100
    print(f"   {c} ({names_9[c]:10s}): {count:>7,} ({pct:>5.1f}%)")

print("\nHorizontal distribution:")
h_names = metadata['h_names']
for h in range(3):
    count = (labels_h == h).sum()
    pct = count / idx * 100
    print(f"   {h} ({h_names[h]:5s}): {count:>7,} ({pct:>5.1f}%)")

print("\nVertical distribution:")
v_names = metadata['v_names']
for v in range(3):
    count = (labels_v == v).sum()
    pct = count / idx * 100
    print(f"   {v} ({v_names[v]:5s}): {count:>7,} ({pct:>5.1f}%)")

print(f"\nFIRE distribution:")
print(f"   No fire: {(labels_fire == 0).sum():>7,} ({(labels_fire == 0).mean()*100:>5.1f}%)")
print(f"   Fire:    {(labels_fire == 1).sum():>7,} ({(labels_fire == 1).mean()*100:>5.1f}%)")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print(f"\nSaved to {OUTPUT_DIR}/")
print(f"   train.npz: {len(train_idx):,} sequences")
print(f"   val.npz: {len(val_idx):,} sequences")
print(f"\nArrays saved:")
print(f"   - sequences: (N, {SEQUENCE_LENGTH}, {IMG_SIZE}, {IMG_SIZE})")
print(f"   - labels_9way: 9-class for baseline")
print(f"   - labels_h: horizontal (LEFT/NONE/RIGHT)")
print(f"   - labels_v: vertical (UP/NONE/DOWN)")
print(f"   - labels_fire: FIRE button pressed")
print(f"   - labels_raw: original action IDs")
