import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class FeatureAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attention(x)

class AdaptiveScaleLayer(nn.Module):
    def __init__(self, dim, mode="input"):
        super().__init__()
        self.mode = mode
        self.mean = nn.Parameter(torch.zeros(dim))
        self.std = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        if self.mode == "input":
            return (x - self.mean) / (self.std)
        elif self.mode == "output":
            return x * (self.std) + self.mean

class AttentionBlock_n(nn.Module):
    def __init__(self, in_dim, out_dim, ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            FeatureAttention(out_dim)
        )
    def forward(self, x):

            return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim,):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            FeatureAttention(out_dim)
        )
    def forward(self, x):
            return self.block(x)


class CGDBNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=512):
        super().__init__()
        self.input_calibrate = AdaptiveScaleLayer(input_dim, mode="input")


        self.feature_extractor = nn.Sequential(
            AttentionBlock_n(input_dim, 512, ),
        )

        self.cl_branch = nn.Sequential(
            AttentionBlock_n(512, 512,),
            AttentionBlock_n(512, 512,),
            AttentionBlock_n(512, 512,),
        )
        self.cl_pred_head = nn.Linear(512, 1)

        self.gate_cl = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.downsample = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.cd_branch = nn.Sequential(
            AttentionBlock(512, 256, ),
            AttentionBlock(256, 256, ),
            AttentionBlock(256, 256, ),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = self.input_calibrate(x)
        feat_shared = self.feature_extractor(x)

        feat_cl_512 = self.cl_branch(feat_shared)
        cl_pred = self.cl_pred_head(feat_cl_512)

        gate = self.gate_cl(feat_cl_512.detach())
        feat_cl_gated = feat_cl_512 * gate

        concat_feat = torch.cat([feat_shared, feat_cl_gated], dim=1)

        fused_feat = self.downsample(concat_feat)

        cd_pred = self.cd_branch(fused_feat)
        return torch.cat([cl_pred, cd_pred], dim=1)