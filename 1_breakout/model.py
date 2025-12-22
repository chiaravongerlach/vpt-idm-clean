#!/usr/bin/env python3
"""

model architecture for the breakout atari simplified vpt MODEL 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=16):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VPTBreakoutIDM(nn.Module):
    """
    Simplified VPT Inverse Dynamics Model for Breakout
    
    Architecture:
    1. 3D Temporal Conv (captures motion)
    2. ResNet-18 Image Encoder (per-frame features)
    3. Transformer Encoder (temporal reasoning)
    4. Action Prediction Head
    """
    def __init__(
        self,
        num_actions=5,
        sequence_length=16,
        img_size=84,
        hidden_dim=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        
        #3d temproal conv
        self.temporal_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1)
        )
        
        # 2d spatial conv
        self.spatial_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.spatial_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.spatial_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.spatial_conv4 = nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(hidden_dim)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # adding positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=sequence_length)
        
        # transformer encode non causal 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # add prediciton head 
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 16, 84, 84) - grayscale frames
        
        Returns:
            logits: (batch, num_actions) - action predictions
        """
        batch_size = x.size(0)
        
        # channel dim
        x = x.unsqueeze(1)  # (batch, 1, 16, 84, 84)
        
        # 1. 3D Temporal Conv
        x = F.relu(self.temporal_conv(x))  # (batch, 32, 16, 84, 84)
        
        # 2. Reshape for 2D spatial processing
        x = x.permute(0, 2, 1, 3, 4)  # (batch, 16, 32, 84, 84)
        x = x.reshape(batch_size * self.sequence_length, 32, self.img_size, self.img_size)
        
        # 2D Spatial Conv layers
        x = F.relu(self.bn1(self.spatial_conv1(x)))
        x = F.relu(self.bn2(self.spatial_conv2(x)))
        x = F.relu(self.bn3(self.spatial_conv3(x)))
        x = F.relu(self.bn4(self.spatial_conv4(x)))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        x = x.view(batch_size * self.sequence_length, self.hidden_dim)
        
        # reshape back to sequences
        x = x.view(batch_size, self.sequence_length, self.hidden_dim)

        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 5. midle frame action pred
        middle_idx = self.sequence_length // 2
        middle_features = x[:, middle_idx, :]
        
        # 6. Predict action
        logits = self.action_head(middle_features)
        
        return logits

def count_parameters(model):
    """number of params trained """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("testing")
    
    model = VPTBreakoutIDM(num_actions=5, sequence_length=16, img_size=84)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 16, 84, 84)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\npassed model test")
