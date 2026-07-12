import torch
from timm.layers import to_2tuple, trunc_normal_
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,patch_size=16,in_channels=3,embed_dim=384):
        super().__init__()
        img_size=to_2tuple(img_size)
        patch_size=to_2tuple(patch_size)
        num_patches=(img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        self.img_size=img_size
        self.patch_size=patch_size
        self.num_patches=num_patches
        self.proj=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        x=self.proj(x)
        x=x.flatten(2).transpose(1,2)
        return x  #（b,196,384）

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x):
        B,N,C=x.shape #（b,197,384）
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]#(b,6,197,64)
        q=q*self.scale
        attn=(q@k.transpose(-2,-1))
        attn=attn.softmax(dim=-1)
        attn=self.attn_drop(attn)

        x=(attn@v).transpose(1,2).reshape(B,N,C) #(b,197,384)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    def forward(self,x):#(b,197,384)
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x

class AirfoilViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=384,
                 depth=6,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 num_features=10,
                 num_y_coords=60):
        super().__init__()
        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))

        self.blocks=nn.ModuleList([
            Block(
                dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                drop=drop_rate,attn_drop=attn_drop_rate,drop_path=drop_path_rate,norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm=norm_layer(embed_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, num_features),
            nn.Sigmoid()
        )
        self.regressor=nn.Sequential(
            nn.Linear(num_features,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256,512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256,num_y_coords)
        )
        trunc_normal_(self.cls_token,std=.02)
        trunc_normal_(self.pos_embed,std=.02)
        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
    def forward_features(self,x):#(b,3,224,224)
        B=x.shape[0]
        x=self.patch_embed(x) #(b,196,384)
        cls_tokens=self.cls_token.expand(B,-1,-1) #(b,1,384)
        x=torch.cat((cls_tokens,x),dim=1)#(b,197,384)
        x=x+self.pos_embed.expand(B,-1,-1)

        for blk in self.blocks:
            x=blk(x)
        x=self.norm(x)#(b,197,384)
        x=x[:,0]
        return x #(b,1,384)
    def forward(self,x):#(b,3,224,224)
        feat=self.forward_features(x)#(b,1,384)
        features=self.feature_proj(feat)#(b,10)
        y_coords=self.regressor(features)#(b,60)
        return features,y_coords

