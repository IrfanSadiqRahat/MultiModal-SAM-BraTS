"""
DualBranchSAM — Paper 1 main model.

Two LoRA-adapted SAM branches (T1 and T2/FLAIR) share frozen backbone
weights but have independent LoRA adapters. Cross-modal attention fuses
features before the segmentation head outputs WT/TC/ET masks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[Warning] Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")

from .lora_adapter import inject_lora_into_vit
from .cross_modal_attention import CrossModalAttention, reshape_sam_features, restore_sam_features


class DualBranchSAM(nn.Module):
    def __init__(self, sam_checkpoint: str, model_type="vit_b", lora_rank=8,
                 lora_alpha=None, num_classes=3, fusion_heads=8, missing_prob=0.15):
        super().__init__()
        if not SAM_AVAILABLE:
            raise ImportError("Install segment-anything first")

        self.num_classes  = num_classes
        self.missing_prob = missing_prob

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder  = sam.image_encoder
        self.mask_decoder   = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder

        # Branch A LoRA injected into encoder
        self.image_encoder = inject_lora_into_vit(
            self.image_encoder, rank=lora_rank, alpha=lora_alpha)

        # Branch B: separate LoRA parameters via forward hooks
        self._build_branch_b_lora(lora_rank, lora_alpha or float(lora_rank))

        embed_dim = 256  # SAM ViT-B neck output channels
        self.fusion = CrossModalAttention(embed_dim=embed_dim, num_heads=fusion_heads)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, num_classes, 1),
        )
        self._print_param_summary()

    def _build_branch_b_lora(self, rank, alpha):
        self.branch_b_lora = nn.ModuleList()
        for block in self.image_encoder.blocks:
            if hasattr(block.attn, "qkv"):
                qkv = block.attn.qkv
                in_f  = qkv.weight.shape[1]
                out_f = qkv.weight.shape[0]
                lora_A = nn.Parameter(torch.empty(rank, in_f))
                lora_B = nn.Parameter(torch.zeros(out_f, rank))
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                self.branch_b_lora.append(nn.ParameterDict({
                    "A": lora_A, "B": lora_B,
                    "scale": nn.Parameter(torch.tensor(alpha/rank), requires_grad=False),
                }))

    def _encode(self, x, use_branch_b=False):
        if not use_branch_b:
            return self.image_encoder(x)
        hooks, idx = [], 0
        for block in self.image_encoder.blocks:
            if hasattr(block.attn, "qkv") and idx < len(self.branch_b_lora):
                p = self.branch_b_lora[idx]
                A, B, sc = p["A"], p["B"], p["scale"]
                def make_hook(A, B, sc):
                    def hook(m, inp, out):
                        return out + sc * (inp[0] @ A.T @ B.T)
                    return hook
                hooks.append(block.attn.qkv.register_forward_hook(make_hook(A, B, sc)))
                idx += 1
        out = self.image_encoder(x)
        for h in hooks: h.remove()
        return out

    def _missing_augment(self, t1, t2f):
        if not self.training or self.missing_prob <= 0:
            return t1, t2f
        for i in range(t1.shape[0]):
            r = torch.rand(1).item()
            if r < self.missing_prob / 2:   t1[i]  = 0.0
            elif r < self.missing_prob:      t2f[i] = 0.0
        return t1, t2f

    def forward(self, t1, t2f, missing_modality=None):
        if missing_modality == "t1":   t1  = torch.zeros_like(t1)
        if missing_modality == "t2f":  t2f = torch.zeros_like(t2f)
        t1, t2f = self._missing_augment(t1, t2f)

        feat_a = self._encode(t1,  use_branch_b=False)
        feat_b = self._encode(t2f, use_branch_b=True)
        H, W   = feat_a.shape[2], feat_a.shape[3]

        fused = restore_sam_features(
            self.fusion(reshape_sam_features(feat_a),
                        reshape_sam_features(feat_b)), H, W)

        up = F.interpolate(fused, size=(256, 256), mode="bilinear", align_corners=False)
        return self.seg_head(up)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nDualBranchSAM | total={total:,} | trainable={trainable:,} ({100*trainable/total:.2f}%)\n")
