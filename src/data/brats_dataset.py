"""
BraTS 2023 Glioma Dataset Loader.

Label conventions:
  WT = {1,2,3}  |  TC = {1,3}  |  ET = {3}
"""
import os, random
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy import ndimage
from PIL import Image


def zscore_normalize(vol: np.ndarray, mask=None) -> np.ndarray:
    vox = vol[mask > 0] if mask is not None else vol[vol > 0]
    if len(vox) == 0: return vol.astype(np.float32)
    return np.clip((vol - vox.mean()) / (vox.std() + 1e-8), -5, 5).astype(np.float32)


def build_brats_masks(seg: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "WT": (seg > 0).astype(np.float32),
        "TC": np.isin(seg, [1, 3]).astype(np.float32),
        "ET": (seg == 3).astype(np.float32),
    }


class BraTS2023Dataset(Dataset):
    def __init__(self, data_root, case_list, slice_axis=2,
                 image_size=1024, min_tumor_voxels=50, augment=False):
        self.data_root        = Path(data_root)
        self.slice_axis       = slice_axis
        self.image_size       = image_size
        self.min_tumor_voxels = min_tumor_voxels
        self.augment          = augment
        self.samples          = self._build_index(case_list)
        print(f"[BraTS2023] {len(case_list)} cases → {len(self.samples)} valid slices")

    def _get_slice(self, vol, idx):
        if self.slice_axis == 0: return vol[idx]
        if self.slice_axis == 1: return vol[:, idx]
        return vol[:, :, idx]

    def _build_index(self, cases):
        samples = []
        for c in cases:
            seg_path = self.data_root / c / f"{c}-seg.nii.gz"
            if not seg_path.exists(): continue
            seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
            for s in range(seg.shape[self.slice_axis]):
                if self._get_slice(seg, s).sum() >= self.min_tumor_voxels:
                    samples.append((self.data_root / c, s))
        return samples

    def _load(self, case_dir, mod):
        p = case_dir / f"{case_dir.name}-{mod}.nii.gz"
        return nib.load(str(p)).get_fdata().astype(np.float32) if p.exists() else None

    def _to_tensor(self, sl):
        sl = np.array(Image.fromarray(sl).resize((self.image_size, self.image_size), Image.BILINEAR))
        mn, mx = sl.min(), sl.max()
        sl = ((sl - mn) / (mx - mn + 1e-8) * 255).astype(np.float32)
        return torch.from_numpy(np.stack([sl, sl, sl]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        case_dir, s_idx = self.samples[idx]
        t1n = self._load(case_dir, "t1n")
        t2f = self._load(case_dir, "t2f")
        seg = nib.load(str(case_dir / f"{case_dir.name}-seg.nii.gz")).get_fdata().astype(np.int16)

        mask = (t1n > 0).astype(np.uint8) if t1n is not None else None
        if t1n is not None: t1n = zscore_normalize(t1n, mask)
        if t2f is not None: t2f = zscore_normalize(t2f, mask)

        H, W   = seg.shape[0], seg.shape[1]
        t1_sl  = self._get_slice(t1n, s_idx) if t1n is not None else np.zeros((H,W))
        t2f_sl = self._get_slice(t2f, s_idx) if t2f is not None else np.zeros((H,W))
        seg_sl = self._get_slice(seg, s_idx)

        if self.augment and random.random() < 0.5:
            t1_sl  = np.fliplr(t1_sl).copy()
            t2f_sl = np.fliplr(t2f_sl).copy()
            seg_sl = np.fliplr(seg_sl).copy()

        masks = build_brats_masks(seg_sl)
        target = torch.from_numpy(np.stack([
            np.array(Image.fromarray(masks[k]).resize((self.image_size, self.image_size), Image.NEAREST))
            for k in ["WT","TC","ET"]
        ], axis=0).astype(np.float32) > 0.5)

        return {
            "t1": self._to_tensor(t1_sl), "t2f": self._to_tensor(t2f_sl),
            "target": target.float(), "case": case_dir.name, "slice": s_idx,
        }


def get_case_list(data_root):
    return sorted(d.name for d in Path(data_root).iterdir()
                  if d.is_dir() and d.name.startswith("BraTS"))


def split_cases(cases, train_frac=0.70, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    s   = cases.copy(); rng.shuffle(s)
    n   = len(s)
    n_t = int(n * train_frac); n_v = int(n * val_frac)
    return s[:n_t], s[n_t:n_t+n_v], s[n_t+n_v:]


def build_dataloaders(data_root, batch_size=4, num_workers=4, image_size=1024, seed=42):
    cases  = get_case_list(data_root)
    tr, va, te = split_cases(cases, seed=seed)
    print(f"[BraTS2023] {len(tr)} train / {len(va)} val / {len(te)} test")
    tr_ds = BraTS2023Dataset(data_root, tr, image_size=image_size, augment=True)
    va_ds = BraTS2023Dataset(data_root, va, image_size=image_size, augment=False)
    te_ds = BraTS2023Dataset(data_root, te, image_size=image_size, augment=False)
    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(te_ds, batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True),
    )
