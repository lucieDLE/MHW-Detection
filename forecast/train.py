import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from forecast.data import make_loaders
from forecast.model import PixelLSTM


def masked_mse(pred: torch.Tensor, target: torch.Tensor, ocean_mask: torch.Tensor) -> torch.Tensor:
    """MSE over ocean cells only. Land should be removed

    pred, target: (B, n_out, H, W)
    ocean_mask:   (1, 1, H, W) — 1 over ocean, 0 over land
    """
    diff2 = (pred - target) ** 2 * ocean_mask
    denom = ocean_mask.sum() * pred.shape[0] * pred.shape[1]
    return diff2.sum() / denom.clamp(min=1.0)


def build_model(name: str, n_in: int, n_out: int) -> nn.Module:
    if name == "lstm":
        return PixelLSTM(n_in=n_in, n_out=n_out)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, ocean_mask: torch.Tensor, device: str, autocast_dtype) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
            pred = model(x)
            loss = masked_mse(pred, y, ocean_mask)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["lstm"], default="lstm")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batchsize", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--n_in", type=int, default=config.DL_N_IN)
    p.add_argument("--n_out", type=int, default=config.DL_N_OUT)
    p.add_argument("--n_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience on val loss")
    p.add_argument("--tag", type=str, default=None, help="Run tag (defaults to timestamp)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_loader, val_loader, _, land_mask = make_loaders(
        n_in=args.n_in,
        n_out=args.n_out,
        batch_size=args.batchsize,
        num_workers=args.n_workers,
    )

    # convert land mask (set to 1) to ocean mask (set land to 0)
    ocean_mask = (~torch.from_numpy(land_mask)).float().to(device).unsqueeze(0).unsqueeze(0)

    model = build_model(args.model, args.n_in, args.n_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optim = AdamW(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    ckpt_dir = ROOT / config.CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model}_{tag}.pt"
    best_path = ckpt_dir / f"{args.model}_best.pt"

    writer = SummaryWriter(log_dir=str(ROOT / config.LOG_DIR / f"{args.model}_{tag}"))

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
                pred = model(x)
                loss = masked_mse(pred, y, ocean_mask)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, ocean_mask, device, autocast_dtype)
        sched.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", sched.get_last_lr()[0], epoch)
        print(f"epoch {epoch+1}: train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "n_in": args.n_in, "n_out": args.n_out, "val_loss": val_loss},
                       best_path)
            print(f"new best, saved: {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save({"model": args.model, "state_dict": model.state_dict(),
                "n_in": args.n_in, "n_out": args.n_out, "val_loss": val_loss},
               ckpt_path)
    writer.close()
    print(f"Done. Best val loss = {best_val:.5f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
