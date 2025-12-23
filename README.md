# Inverse Dynamics Modeling for Atari: A Factorized Action Head Approach

Repository contains code used for the Breakout experiment, and Factorized vs 9-way Ablation Study for Inverse Dynamics Models for the MsPacman dataset. 

## Overview

Implemented and compared two approaches for predicting actions from video:
1. **Standard 9-way classification** - predicts one of 9 joystick actions
2. **Factorized classification** - predicts horizontal (L/N/R) and vertical (U/N/D) components independently

## Key Findings

| Model | Macro-F1 | Balanced Acc | Notes |
|-------|----------|--------------|-------|
| 9-Way Baseline | **0.632** | 0.624 | Standard classification |
| Factorized | 0.615 | **0.633** | Better class balance, interpretable |

## Repository Structure
```
├── 1_breakout/          
│   ├── download.py      # download Atari HEAD dataset for Breakout
│   ├── preprocess.py    # create training + validation sequences
│   ├── train.py         # Train IDM
│   └── results/         
│
├── 2_mspacman/          # Main experiments (factorized approach)
│   ├── download.py      # Download Atari-HEAD Ms. Pacman data
│   ├── preprocess.py    # creating training and validation sequences
│   ├── train.py         # ablation training
│   ├── checkpoints/     # model weights
│   └── results/         # three version
```

## Requirements
```
torch>=2.0
numpy
scikit-learn
matplotlib
tqdm
```

## References

Motivated by OpenAI's Video Pre-Training (VPT) methodology and using the Atari-HEAD dataset.
