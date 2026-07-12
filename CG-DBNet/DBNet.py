import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class AdaptiveScaleLayer(nn.Module):
    def __init__(self, dim, mode="input"):
        super().__init__()
        self.mode = mode
        self.mean = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.std = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        if self.mode == "input":
            return (x - self.mean) / (self.std)

class AttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
            return self.block(x)

class DBNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=512):
        super().__init__()
        self.input_calibrate = AdaptiveScaleLayer(input_dim, mode="input")
        self.feature_extractor = nn.Sequential(
            AttentionBlock(input_dim, 512, ),
        )
        self.cl_branch = nn.Sequential(
            AttentionBlock(512, 512,),
            AttentionBlock(512, 512,),
            AttentionBlock(512, 512, ),
            nn.Linear(512, 1)
        )

        self.cd_branch = nn.Sequential(
            AttentionBlock(512, 256, ),
            AttentionBlock(256, 256, ),
            AttentionBlock(256, 256, ),
            nn.Linear(256, 1)
        )

    def forward(self, x):

        x = self.input_calibrate(x)
        feat = self.feature_extractor(x)
        cl_pred = self.cl_branch(feat)
        cd_pred = self.cd_branch(feat)

        pred = torch.cat([cl_pred, cd_pred], dim=1)
        return pred



