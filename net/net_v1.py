import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, in_k=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, in_k=in_k)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, k=None):
        x = x + self.drop_path(self.attn(self.norm1(x), k))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., in_k=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.in_k = in_k
        self.qkv = nn.Linear(dim, dim * 2 if in_k else dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, in_k=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 2 if self.in_k else 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if self.in_k and in_k is not None:
            q, k, v = qkv[0], in_k, qkv[1]
            B, N, C = x.shape
            k = k.reshape(B, self.num_heads, N, -1)
        else:
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_chans=3, num_features=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class init(nn.Module):
    def __init__(self, input_shape=[8, 8], patch_size=1, in_chans=144, num_features=384, drop_rate=0.1):
        super().__init__()
        self.input_shape = input_shape
        self.patch_embed_1 = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                        num_features=num_features)
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.num_features = num_features
        self.new_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.patch_embed_1(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        cls_token_pe = self.pos_embed[:, 0:1, :]
        img_token_pe = self.pos_embed[:, 1:, :]

        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        x = self.pos_drop(x + pos_embed)
        return x


class build(nn.Module):
    def __init__(self, input_shape=[16, 16], patch_size=1, in_chans_hsi=144, in_chans_lidar=1, num_classes=1000,
                 num_features=384,
                 depth=6, num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1, hot_map=False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU, return_feature=False, input_stack=False
                 ):
        super().__init__()
        self.return_feature = return_feature
        self.input_stack = input_stack
        self.input_shape = input_shape
        self.hot_map = hot_map
        self.num_features = num_features
        self.in_chans_hsi = in_chans_hsi
        self.hsi_init = init(
            input_shape, patch_size, in_chans_hsi, num_features, drop_rate
        )
        self.lidar_init = init(
            input_shape, patch_size, in_chans_lidar, num_features, drop_rate
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_hsi = nn.Sequential(
            *[
                Block(
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    in_k=False if i == 0 else True
                ) for i in range(depth)
            ]
        )
        self.blocks_lidar = copy.deepcopy(self.blocks_hsi)
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                ) for i in range(depth)
            ]
        )
        self.norm = norm_layer(num_features)
        self.norm1 = norm_layer(num_features)
        self.norm2 = norm_layer(num_features)

        self.head = nn.Conv2d(num_features, num_classes, 1, 1) if num_classes > 0 else nn.Identity()
        self.head1 = nn.Conv2d(num_features, num_classes, 1, 1) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Conv2d(num_features, num_classes, 1, 1) if num_classes > 0 else nn.Identity()

    def forward(self, x, y=None):
        if self.input_stack:
            x, y = x[:, :self.in_chans_hsi], x[:, self.in_chans_hsi:]
        x = self.hsi_init(x)
        y = self.lidar_init(y)

        for i, (block_hsi, block_lidar, block) in enumerate(zip(self.blocks_hsi, self.blocks_lidar, self.blocks)):
            if i == 0:
                x = block_hsi(x)
                y = block_lidar(y)
            else:
                x = block_hsi(x, z)
                y = block_lidar(y, z)
            z = block(x + y)
        x = self.norm(x)[:, 1:]
        y = self.norm1(y)[:, 1:]
        z = self.norm2(z)[:, 1:]

        x = x.reshape((-1, *self.input_shape, self.num_features)).permute(0, 3, 1, 2)
        y = y.reshape((-1, *self.input_shape, self.num_features)).permute(0, 3, 1, 2)
        z = z.reshape((-1, *self.input_shape, self.num_features)).permute(0, 3, 1, 2)
        hsi_out = self.head(x)
        lidar_out = self.head1(y)
        out = self.head2(z)
        if self.return_feature:
            return hsi_out, lidar_out, out, z
        if self.hot_map:
            return out
        return hsi_out, lidar_out, out


if __name__ == '__main__':
    x1 = torch.randn(size=(2, 144, 16, 16))
    x2 = torch.randn(size=(2, 1, 16, 16))

    net = build(num_classes=16)
    y = net(x1, x2)
