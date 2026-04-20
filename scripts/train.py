"""
Main training script.
Usage: python scripts/train.py --config configs/brats2023.yaml
"""
import sys, argparse, yaml, time
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.dual_branch_sam import DualBranchSAM
from src.data.brats_dataset import build_dataloaders
from src.training.losses import CombinedSegLoss
from src.evaluation.metrics import SegmentationMetrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/brats2023.yaml")
    p.add_argument("--sam_ckpt",   default="checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--output_dir", default="outputs/run_001")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--resume",     default=None)
    p.add_argument("--seed",       type=int, default=42)
    # CLI overrides
    p.add_argument("--lora_rank",  type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    return p.parse_args()


def train_epoch(model, loader, opt, crit, device, epoch):
    model.train(); total = 0
    for i, batch in enumerate(loader):
        t1, t2f, tgt = batch["t1"].to(device), batch["t2f"].to(device), batch["target"].to(device)
        opt.zero_grad()
        d = crit(model(t1, t2f), tgt)
        d["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += d["loss"].item()
        if i % 50 == 0:
            print(f"  [{epoch}][{i}/{len(loader)}] loss={d['loss']:.4f} "
                  f"dice={d['dice_loss']:.3f} bce={d['bce_loss']:.3f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, crit, device, mets):
    model.eval(); mets.reset(); total = 0
    for batch in loader:
        t1, t2f, tgt = batch["t1"].to(device), batch["t2f"].to(device), batch["target"].to(device)
        out = model(t1, t2f)
        total += crit(out, tgt)["loss"].item()
        mets.update(torch.sigmoid(out).cpu().numpy(), tgt.cpu().numpy())
    return total / max(len(loader), 1), mets.compute()


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    if args.lora_rank:   cfg["lora_rank"]  = args.lora_rank
    if args.batch_size:  cfg["batch_size"] = args.batch_size
    if args.epochs:      cfg["epochs"]     = args.epochs
    if args.lr:          cfg["lr"]         = args.lr

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out    = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Config: {cfg}")

    tr_loader, va_loader, _ = build_dataloaders(
        cfg["data_root"], cfg.get("batch_size",4),
        cfg.get("num_workers",4), cfg.get("image_size",1024), args.seed)

    model = DualBranchSAM(
        sam_checkpoint=args.sam_ckpt, lora_rank=cfg.get("lora_rank",8),
        num_classes=3, fusion_heads=cfg.get("fusion_heads",8),
        missing_prob=cfg.get("missing_prob",0.15),
    ).to(device)

    opt   = AdamW(model.get_trainable_params(), lr=cfg.get("lr",1e-4), weight_decay=cfg.get("weight_decay",1e-4))
    sched = CosineAnnealingLR(opt, T_max=cfg.get("epochs",100))
    crit  = CombinedSegLoss(); mets = SegmentationMetrics(compute_hd95=False) if False else SegmentationMetrics()

    best, start = 0.0, 1
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start = ck["epoch"]+1; best = ck.get("best_dice",0.0)
        print(f"Resumed from epoch {ck['epoch']}")

    for epoch in range(start, cfg.get("epochs",100)+1):
        t0 = time.time()
        tr_loss        = train_epoch(model, tr_loader, opt, crit, device, epoch)
        va_loss, va_m  = validate(model, va_loader, crit, device, mets)
        sched.step()
        dice = va_m["DICE_mean"]
        print(f"Epoch {epoch:3d} | tr={tr_loss:.4f} va={va_loss:.4f} | "
              f"DICE mean={dice:.4f} WT={va_m['DICE_WT']:.4f} TC={va_m['DICE_TC']:.4f} ET={va_m['DICE_ET']:.4f} | "
              f"{time.time()-t0:.1f}s")
        if dice > best:
            best = dice
            torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),
                        "best_dice":best,"config":cfg}, out/"best_model.pth")
            print(f"  ✅ New best DICE={best:.4f} saved")
        if epoch % cfg.get("save_every",10) == 0:
            torch.save({"epoch":epoch,"model":model.state_dict(),
                        "optimizer":opt.state_dict()}, out/f"ckpt_{epoch:03d}.pth")

    print(f"\nTraining complete. Best DICE: {best:.4f}")

if __name__ == "__main__": main()
