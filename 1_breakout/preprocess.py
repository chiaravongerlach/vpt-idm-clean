import os
import tarfile
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
import pickle

breakout_dir = "breakout"
output_dir = "breakout_processed"
os.makedirs(output_dir, exist_ok=True)

# Atari Breakout action mapping
ACTION_MAP = {
    0: 0,   # NOOP → NOOP
    1: 1,   # FIRE → FIRE
    2: 0,   # UP → NOOP (unused)
    3: 2,   # RIGHT → RIGHT
    4: 3,   # LEFT → LEFT
    5: 0,   # DOWN → NOOP (unused)
    11: 3,  # LEFTFIRE → LEFT
    12: 2,  # RIGHTFIRE → RIGHT
}
NUM_ACTIONS = 4

print(f"\nAction Mapping:")
for k, v in ACTION_MAP.items():
    print(f"   Atari action {k} → Model action {v}")

# Get all trials - files end with .tar.bz2
regular_trials = [f.replace('.tar.bz2', '') for f in os.listdir(breakout_dir) if f.endswith('.tar.bz2')]
highscore_trials = [f.replace('.tar.bz2', '') for f in os.listdir(os.path.join(breakout_dir, 'highscore')) if f.endswith('.tar.bz2')]

print(f"\nFound {len(regular_trials)} regular + {len(highscore_trials)} highscore trials")

all_sequences = []
action_counts = Counter()

# Process first 3 trials for quick testing
trials_to_process = regular_trials[:3]

for trial_name in trials_to_process:
    print(f"\nProcessing {trial_name}...")
    
    # Paths
    tar_path = os.path.join(breakout_dir, f"{trial_name}.tar.bz2")
    txt_path = os.path.join(breakout_dir, f"{trial_name}.txt")
    frames_dir = os.path.join(breakout_dir, trial_name)
    
    # Extract if not already extracted
    if not os.path.exists(frames_dir):
        print(f"   Extracting frames...")
        with tarfile.open(tar_path, 'r:bz2') as tar:
            tar.extractall(path=breakout_dir)
    else:
        print(f"   Frames already extracted")
    
    # Load actions
    print(f"   Loading actions...")
    actions = []
    with open(txt_path, 'r') as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > 5:
                try:
                    action = int(parts[5])
                    actions.append(ACTION_MAP.get(action, 0))  # Map action
                except:
                    actions.append(0)  # Default to NOOP if parse fails
    
    print(f"   Total frames: {len(actions)}")
    action_counts.update(actions)
    
    # Create sequences (skip for now to save time)
    num_sequences = len(actions) - 16
    all_sequences.append((trial_name, len(actions), num_sequences))

print(f"\nSUMMARY:")
print(f"   Trials processed: {len(trials_to_process)}")
print(f"   Total frames: {sum([s[1] for s in all_sequences]):,}")
print(f"   Total sequences: {sum([s[2] for s in all_sequences]):,}")

print(f"\na ction Distribution:")
total = sum(action_counts.values())
for action in sorted(action_counts.keys()):
    count = action_counts[action]
    pct = (count / total) * 100
    print(f"   Action {action}: {count:,} ({pct:.1f}%)")

