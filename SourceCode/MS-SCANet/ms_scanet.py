import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, trunc_normal_
from functools import partial, reduce
import operator
from typing import Tuple
import config
import torch.nn.functional as F


# Utility functions
def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x


class SinusoidalPosEnc(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.encoding = self.create_encoding(embed_dim, max_seq_len=768)

    def create_encoding(self, embed_dim, max_seq_len):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(max_seq_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        # Assume x is of shape [B, N, C], where N could be different from max_seq_len
        return x + self.encoding[:x.size(1), :].to(x.device)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=256, max_seq_len=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.positional_encoding = SinusoidalPosEnc(embed_dim, max_seq_len)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."

        # Apply the projection to get the patch embeddings
        x = self.proj(x)  # B, E, H', W'

        # Flatten the embeddings and transpose to get the shape B, N, E
        x = x.flatten(2).transpose(1, 2)  # B, E, (H'xW')

        # Add positional encoding
        x = self.positional_encoding(x)

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SpatialBlock, self).__init__()
        self.norm = norm_layer(dim)
        self.window_attention = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                                attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        # def __init__(
        #         self,
        #         in_features,
        #         hidden_features=None,
        #         out_features=None,
        #         act_layer=nn.GELU):

    def forward(self, x):
        x = self.norm(x)
        x = self.window_attention(x)
        x = self.drop_path(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.Conv2d(embed_dim, embed_dim // squeeze_factor, 1)  # Squeeze using 1x1 conv
        self.excite = nn.Conv2d(embed_dim // squeeze_factor, embed_dim, 1)  # Excitation using 1x1 conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, D = x.shape
        # Reshape x to [B*N, D, 1, 1] to treat each patch separately
        x_reshaped = x.view(B * N, D, 1, 1)

        # Squeeze and excitation
        y = self.squeeze(x_reshaped)
        y = self.excite(y)
        y = self.sigmoid(y)

        # Reshape back to [B, N, D] for broadcasting
        y = y.view(B, N, D)

        # Element-wise multiplication of x and y
        x = x * y
        return x


class ChannelBlock(nn.Module):
    """Channel-wise attention block with single-head attention and optional MLP."""

    def __init__(self, dim, mlp_ratio, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 squeeze_factor=16):
        super(ChannelBlock, self).__init__()
        self.norm = norm_layer(dim)
        self.channel_attention = ChannelAttention(dim, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        x = self.norm(x)
        x = self.channel_attention(x)
        x = self.drop_path(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x


class CrossBranchAttention(nn.Module):
    """Cross-attention allowing branches to attend to each other's features with variable dimensions."""

    def __init__(self, embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Separate QKV projections for x and y
        self.qkv_x = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        # Projection layers back to their original dimensions
        self.proj_x = nn.Linear(embed_dim, embed_dim)
        self.proj_y = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        B, N, _ = x.shape
        _, M, _ = y.shape

        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, self.embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_y = self.qkv_y(y).reshape(B, M, 3, self.num_heads, self.embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q_x, k_x, v_x = qkv_x.unbind(0)  # Queries, keys, values for x
        q_y, k_y, v_y = qkv_y.unbind(0)  # Queries, keys, values for y

        # Cross-attention where x attends to y and vice versa
        attn_xy = (q_x @ k_y.transpose(-2, -1)) * (1.0 / (self.embed_dim // self.num_heads) ** 0.5)
        attn_yx = (q_y @ k_x.transpose(-2, -1)) * (1.0 / (self.embed_dim // self.num_heads) ** 0.5)

        attn_xy = self.softmax(attn_xy)
        attn_yx = self.softmax(attn_yx)

        attn_xy = self.attn_drop(attn_xy)
        attn_yx = self.attn_drop(attn_yx)

        # Apply attention and add to the original inputs
        x = x + self.proj_drop(self.proj_x((attn_xy @ v_y).transpose(1, 2).reshape(B, N, self.embed_dim)))
        y = y + self.proj_drop(self.proj_y((attn_yx @ v_x).transpose(1, 2).reshape(B, M, self.embed_dim)))

        return x, y


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MultiScaleDualAttentionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_sizes=(16, 32), in_chans=3, embed_dim=256, num_heads=8, window_size=4,
                 mlp_ratio=4, qkv_bias=True, attn_drop_rate=0., drop_rate=0., drop_path_rate=0., num_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes

        self.patch_embed1 = PatchEmbed(img_size, patch_size=patch_sizes[0], in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed2 = PatchEmbed(img_size, patch_size=patch_sizes[1], in_chans=in_chans, embed_dim=embed_dim)

        self.num_patches1 = (img_size[0] // patch_sizes[0]) * (img_size[1] // patch_sizes[0])
        self.num_patches2 = (img_size[0] // patch_sizes[1]) * (img_size[1] // patch_sizes[1])
        self.pos_enc1 = SinusoidalPosEnc(embed_dim, max_seq_len=self.num_patches1)
        self.pos_enc2 = SinusoidalPosEnc(embed_dim, max_seq_len=self.num_patches2)

        self.adaptive_pool_size = (img_size[0] // patch_sizes[0], img_size[1] // patch_sizes[0])
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.adaptive_pool_size)

        self.branch1_blocks = nn.ModuleList([
            ChannelBlock(embed_dim, mlp_ratio, drop=drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
                         norm_layer=nn.LayerNorm, squeeze_factor=16),
            SpatialBlock(embed_dim, num_heads, window_size, mlp_ratio, qkv_bias=qkv_bias, drop=attn_drop_rate,
                         attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
                         norm_layer=nn.LayerNorm),
        ])
        self.branch2_blocks = nn.ModuleList([
            ChannelBlock(embed_dim, mlp_ratio, drop=drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
                         norm_layer=nn.LayerNorm, squeeze_factor=16),
            SpatialBlock(embed_dim, num_heads, window_size, mlp_ratio, qkv_bias=qkv_bias, drop=attn_drop_rate,
                         attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=nn.GELU,
                         norm_layer=nn.LayerNorm),
        ])
        self.cross_branch_attn = CrossBranchAttention(embed_dim, num_heads, qkv_bias=qkv_bias, attn_drop=0.,
                                                      proj_drop=0.)
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.prediction_head = Mlp(in_features=self.num_patches1 * embed_dim * 2, hidden_features=embed_dim * 2, out_features=1,
                        act_layer=nn.GELU)  # Predicted value
        self.apply(_init_weights)

    def forward(self, x, return_embeddings=False):
        x1 = self.patch_embed1(x)
        x2 = self.patch_embed2(x)
        x1 = self.pos_enc1(x1)
        x2 = self.pos_enc2(x2)

        B, N2, E = x2.shape
        H2, W2 = int(N2 ** 0.5), int(N2 ** 0.5)
        x2_reshaped = x2.transpose(1, 2).view(B, E, H2, W2)
        x2_pooled = self.adaptive_pool(x2_reshaped)
        x2 = x2_pooled.view(B, E, -1).transpose(1, 2)

        for block in self.branch1_blocks:
            x1 = block(x1)
        for block in self.branch2_blocks:
            x2 = block(x2)

        x1, x2 = self.cross_branch_attn(x1, x2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.norm(x)
        embeddings = x.view(x.size(0), -1)

        if return_embeddings:
            return embeddings

        prediction = self.prediction_head(embeddings)
        return prediction, x1, x2, x2_reshaped, x2_pooled



def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


model = MultiScaleDualAttentionTransformer(
    img_size=(224, 224),  # Example input image size
    patch_sizes=[16, 32],  # Example patch sizes
    in_chans=3,  # Number of input channels (e.g., RGB)
    embed_dim=256,  # Embedding dimensions for each scale
    num_heads=8,  # Number of attention heads for each scale
    mlp_ratio=4,  # MLP ratios for each scale
    qkv_bias=True,  # Whether to include bias in the QKV projections
    drop_rate=0.,  # Dropout rate
    attn_drop_rate=0.,  # Attention dropout rate
    drop_path_rate=0.  # DropPath rate
)
