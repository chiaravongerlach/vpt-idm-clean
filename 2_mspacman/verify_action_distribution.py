# download and first verify action distribution for ms pacman 

import os
from collections import Counter

ms_pacman_dir = "ms_pacman/ms_pacman"

# Find all txt files (action labels)
txt_files = []
for root, dirs, files in os.walk(ms_pacman_dir):
    for f in files:
        if f.endswith('.txt'):
            txt_files.append(os.path.join(root, f))

print(f"Found {len(txt_files)} trial files")

# Count all actions across all trials
all_actions = Counter()
total_frames = 0

for txt_file in txt_files:
    try:
        with open(txt_file, 'r') as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 5:
                    try:
                        action = int(parts[5])
                        all_actions[action] += 1
                        total_frames += 1
                    except:
                        pass
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")

print(f"\nTotal frames: {total_frames:,}")
print(f"\nMS. PACMAN ACTION DISTRIBUTION:")
print("=" * 50)

# Ms. Pacman action mapping (9 actions)
action_names = {
    0: 'NOOP',
    1: 'UP', 
    2: 'RIGHT',
    3: 'LEFT',
    4: 'DOWN',
    5: 'UPRIGHT',
    6: 'UPLEFT',
    7: 'DOWNRIGHT',
    8: 'DOWNLEFT'
}

for action in sorted(all_actions.keys()):
    count = all_actions[action]
    pct = (count / total_frames) * 100
    name = action_names.get(action, f'UNKNOWN_{action}')
    print(f"   {action} ({name:10s}): {count:>8,} ({pct:>5.1f}%)")

# Calculate NOOP percentage
noop_pct = (all_actions.get(0, 0) / total_frames) * 100
print(f"\nKEY METRIC:")
print(f"   NOOP percentage: {noop_pct:.1f}%")
if noop_pct < 50:
    print("   MUCH BETTER than Breakout (78% NOOP)!")
else:
    print("   Still imbalanced but better than Breakout")