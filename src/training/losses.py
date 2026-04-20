"""Combined Dice + BCE + Focal loss for brain tumor segmentation."""
import torch, torch.nn as nn, torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__(); self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        B, C = p.shape[:2]
        p = p.view(B,C,-1); t = targets.view(B,C,-1)
        i = (p*t).sum(-1); u = p.sum(-1) + t.sum(-1)
        return (1 - (2*i + self.smooth) / (u + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        ce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.sigmoid(logits)*targets + (1-torch.sigmoid(logits))*(1-targets)
        return (self.alpha * (1-p_t)**self.gamma * ce).mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, dice_w=0.6, bce_w=0.2, focal_w=0.2):
        super().__init__()
        self.w_d=dice_w; self.w_b=bce_w; self.w_f=focal_w
        self.dice=DiceLoss(); self.focal=FocalLoss()

    def forward(self, logits, targets):
        d = self.dice(logits, targets)
        b = F.binary_cross_entropy_with_logits(logits, targets)
        f = self.focal(logits, targets)
        return {"loss": self.w_d*d + self.w_b*b + self.w_f*f,
                "dice_loss": d.item(), "bce_loss": b.item(), "focal_loss": f.item()}
