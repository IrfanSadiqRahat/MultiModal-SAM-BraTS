"""DICE, HD95, Sensitivity, Specificity for BraTS evaluation."""
import numpy as np
from typing import Dict, List


def dice_coefficient(pred, target, smooth=1e-6):
    p = (pred > 0.5).flatten().astype(np.float32)
    t = (target > 0.5).flatten().astype(np.float32)
    return float((2*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth))


def sensitivity(pred, target, smooth=1e-6):
    p = (pred>0.5).flatten().astype(np.float32)
    t = (target>0.5).flatten().astype(np.float32)
    return float(((p*t).sum()+smooth) / (t.sum()+smooth))


def specificity(pred, target, smooth=1e-6):
    p = (pred>0.5).flatten().astype(np.float32)
    t = (target>0.5).flatten().astype(np.float32)
    tn = ((1-p)*(1-t)).sum(); fp = (p*(1-t)).sum()
    return float((tn+smooth)/(tn+fp+smooth))


class SegmentationMetrics:
    CLASS_NAMES = ["WT","TC","ET"]

    def __init__(self, class_names=None):
        self.class_names = class_names or self.CLASS_NAMES
        self.reset()

    def reset(self):
        self.dice = {c:[] for c in self.class_names}
        self.sens = {c:[] for c in self.class_names}
        self.spec = {c:[] for c in self.class_names}
        self.n = 0

    def update(self, preds, targets):
        B, C = preds.shape[:2]
        for b in range(B):
            for ci, cn in enumerate(self.class_names):
                if ci >= C: continue
                p, t = preds[b,ci], targets[b,ci]
                self.dice[cn].append(dice_coefficient(p,t))
                self.sens[cn].append(sensitivity(p,t))
                self.spec[cn].append(specificity(p,t))
            self.n += 1

    def compute(self):
        r = {"n_samples": self.n}
        for c in self.class_names:
            r[f"DICE_{c}"]     = float(np.mean(self.dice[c])) if self.dice[c] else 0.0
            r[f"DICE_{c}_std"] = float(np.std(self.dice[c]))  if self.dice[c] else 0.0
            r[f"Sens_{c}"]     = float(np.mean(self.sens[c])) if self.sens[c] else 0.0
            r[f"Spec_{c}"]     = float(np.mean(self.spec[c])) if self.spec[c] else 0.0
        r["DICE_mean"] = float(np.mean([r[f"DICE_{c}"] for c in self.class_names]))
        return r

    def summary(self):
        r = self.compute()
        lines = [f"n={r['n_samples']} | mean DICE={r['DICE_mean']:.4f}"]
        for c in self.class_names:
            lines.append(f"  {c}: DICE={r[f'DICE_{c}']:.4f}±{r[f'DICE_{c}_std']:.4f} "
                         f"Sens={r[f'Sens_{c}']:.4f} Spec={r[f'Spec_{c}']:.4f}")
        return "\n".join(lines)
