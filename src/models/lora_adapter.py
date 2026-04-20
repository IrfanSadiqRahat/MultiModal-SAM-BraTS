"""
LoRA (Low-Rank Adaptation) for SAM ViT encoder.
Injects trainable low-rank matrices into QKV projections.
All original SAM weights remain frozen.
Reference: Hu et al. (2022) https://arxiv.org/abs/2106.09685
"""
import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """Frozen Linear + trainable low-rank delta: W + scale * (B @ A)"""

    def __init__(self, original_layer: nn.Linear, rank: int = 8,
                 alpha: Optional[float] = None, dropout: float = 0.0):
        super().__init__()
        self.in_features  = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank  = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scale = self.alpha / self.rank

        self.weight = original_layer.weight
        self.bias   = original_layer.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = nn.functional.linear(x, self.weight, self.bias)
        delta = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + self.scale * delta

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, scale={self.scale:.3f}"


def inject_lora_into_vit(vit_encoder, rank=8, alpha=None, target_layers=None):
    """Freeze ViT encoder and inject LoRA into attention QKV projections."""
    for param in vit_encoder.parameters():
        param.requires_grad_(False)

    blocks  = vit_encoder.blocks
    layers  = target_layers if target_layers else list(range(len(blocks)))
    adapted = 0
    for idx in layers:
        block = blocks[idx]
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank,
                                         alpha=alpha or float(rank))
            adapted += 1

    total     = sum(p.numel() for p in vit_encoder.parameters())
    trainable = sum(p.numel() for p in vit_encoder.parameters() if p.requires_grad)
    print(f"[LoRA] {adapted} blocks adapted | trainable={trainable:,}/{total:,} "
          f"({100*trainable/total:.2f}%)")
    return vit_encoder
