"""
Cross-Modal Attention Fusion for T1 and T2/FLAIR feature maps.

Bidirectional cross-attention: T1 attends to T2/FLAIR and vice versa,
then features are fused via average + lightweight MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    F_a attends to F_b, F_b attends to F_a.
    Output: average of updated features passed through fusion MLP.

    Args:
        embed_dim: feature dimension (256 for SAM ViT-B)
        num_heads: number of attention heads
        dropout:   attention dropout
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        for branch in ("a", "b"):
            setattr(self, f"norm_{branch}",    nn.LayerNorm(embed_dim))
            setattr(self, f"q_proj_{branch}",  nn.Linear(embed_dim, embed_dim, bias=False))
            setattr(self, f"k_proj_{branch}",  nn.Linear(embed_dim, embed_dim, bias=False))
            setattr(self, f"v_proj_{branch}",  nn.Linear(embed_dim, embed_dim, bias=False))
            setattr(self, f"out_proj_{branch}", nn.Linear(embed_dim, embed_dim, bias=False))

        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_mlp  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _attend(self, q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj):
        B, N, C = q_feat.shape
        Q = q_proj(q_feat).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        K = k_proj(kv_feat).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        V = v_proj(kv_feat).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)
        attn = F.softmax((Q @ K.transpose(-2,-1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        return out_proj((attn @ V).transpose(1,2).reshape(B, N, C))

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_a: (B, N, C) — T1 features
            feat_b: (B, N, C) — T2/FLAIR features
        Returns:
            fused:  (B, N, C)
        """
        a_n = self.norm_a(feat_a)
        b_n = self.norm_b(feat_b)
        feat_a = feat_a + self._attend(a_n, b_n, self.q_proj_a, self.k_proj_a,
                                        self.v_proj_a, self.out_proj_a)
        feat_b = feat_b + self._attend(b_n, a_n, self.q_proj_b, self.k_proj_b,
                                        self.v_proj_b, self.out_proj_b)
        fused = (feat_a + feat_b) * 0.5
        return fused + self.fusion_mlp(self.fusion_norm(fused))


def reshape_sam_features(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, H*W, C)"""
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)


def restore_sam_features(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, H*W, C) → (B, C, H, W)"""
    B, N, C = x.shape
    return x.transpose(1, 2).reshape(B, C, h, w)
