
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from forecast.baselines import persistence_forecast
from forecast.data import SSTADataset
from forecast.model import PixelLSTM
from forecast.rollout import autoregressive_rollout

LEAD_TIMES = (1, 3, 7, 14)


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt["model"]
    n_in, n_out = ckpt["n_in"], ckpt["n_out"]
    model = PixelLSTM(n_in=n_in, n_out=n_out)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, n_in, n_out


def rmse_acc(pred: np.ndarray, truth: np.ndarray, ocean: np.ndarray) -> tuple[float, float]:
    p = pred[:, ocean]
    t = truth[:, ocean]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    num = float(np.sum(p * t))
    den = float(np.sqrt(np.sum(p ** 2) * np.sum(t ** 2)) + 1e-12)
    acc = num / den
    return rmse, acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=config.DL_HORIZON)
    p.add_argument("--batchsize", type=int, default=1)
    # p.add_argument("--mhw", help="compute MHW detection classification")
    p.add_argument("--out", type=Path, default=ROOT / "forecast" / "eval_results.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, n_in, n_out = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint.name}  n_in={n_in}  n_out={n_out}")

    test_ds = SSTADataset(
        config.SSTA_DAILY_PATH,
        config.LANDMASK_PATH,
        config.DL_TEST_RANGE,
        n_in=n_in,
        n_out=args.horizon, # the number of prediction days we want, not the days the model was trained with
    )
    ocean = ~test_ds.land_mask
    H, W = ocean.shape

    # Accumulate per-lead-time predictions/truths for each method
    methods = ["model", "persistence"]
    accum = {m: {k: {"sse": 0.0, "n": 0, "acc_numerator": 0.0, "acc_denom_pred": 0.0, "acc_denom_target": 0.0} for k in LEAD_TIMES}
             for m in methods} # where sse: Sum of Squared Errors and n: number of sample

    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=2)
    ocean_t = torch.from_numpy(ocean).to(device)

    for x, y in tqdm(loader, desc="evaluating"):
        x = x.to(device)
        y = y.to(device)

        preds = {
            "model":       autoregressive_rollout(model, x, args.horizon, n_out=n_out),
            "persistence": persistence_forecast(x, args.horizon),
        }
        for m, pred in preds.items():
            #we are adding little by little all values needed to compute rmse and accuracy
            for k in LEAD_TIMES: 
                pred_k = pred[:, k - 1][:, ocean_t]
                target_k = y[:, k - 1][:, ocean_t]
                a = accum[m][k]
                a["sse"] += float(((pred_k - target_k) ** 2).sum().item())
                a["n"]   += pred_k.numel()
                a["acc_numerator"] += float((pred_k * target_k).sum().item())
                a["acc_denom_pred"] += float((pred_k ** 2).sum().item())
                a["acc_denom_target"] += float((target_k ** 2).sum().item())

    rows = []
    print(f"\n{'method':<14}{'lead':>6}{'RMSE':>10}{'ACC':>10}")
    for m in methods:
        for k in LEAD_TIMES:
            a = accum[m][k]
            rmse = (a["sse"] / max(a["n"], 1)) ** 0.5
            acc = a["acc_numerator"] / ((a["acc_denom_pred"] * a["acc_denom_target"]) ** 0.5 + 1e-12)
            rows.append((m, k, rmse, acc))
            print(f"{m:<14}{k:>6}{rmse:>10.4f}{acc:>10.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("method,lead_days,rmse,acc\n")
        for m, k, r, a in rows:
            f.write(f"{m},{k},{r:.6f},{a:.6f}\n")
    print(f"\nResults → {args.out}")



if __name__ == "__main__":
    main()
