
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from forecast.baselines import persistence_forecast, RidgeBaseline
from forecast.data import SSTADataset
from forecast.model import build_model
from forecast.rollout import autoregressive_rollout
import matplotlib.pyplot as plt
import os
LEAD_TIMES = (1, 3, 7, 14)

def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = .5 if normalize else np.sum(cm)/4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    return cm

def reduce_factor(ds, target_shape):
    if ds.dims['lat'] != target_shape[1]:
        factor =  int(ds.dims['lat'] / target_shape[1])
        ds = ds.coarsen(lon=factor, lat=factor, boundary='trim').mean()
    return ds

def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt["model"]
    n_in, n_out = ckpt["n_in"], ckpt["n_out"]
    model = build_model(name, n_in=n_in, n_out=n_out)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, n_in, n_out


def spatial_acc_batch(pred_ocean: np.ndarray, truth_ocean: np.ndarray) -> np.ndarray:
    """Spatial Pearson correlation for each sample in a batch.

    The Anomaly Correlation Coefficient (ACC)is  used in SST forecasting: for each timestep,
    compute the Pearson correlation of the predicted vs true field over all ocean
    pixels, then average those per-timestep values over the test period.

    Measure of the relationship between predicted and truth while taking in account their location

    pred_ocean, truth_ocean: (B, n_ocean)
    returns: (B,) per-sample spatial correlations

    see https://en.wikipedia.org/wiki/Lee%27s_L
    """
    p = pred_ocean  - pred_ocean.mean( axis=1, keepdims=True)
    t = truth_ocean - truth_ocean.mean(axis=1, keepdims=True)
    num = (p * t).sum(axis=1)
    den = np.sqrt((p ** 2).sum(axis=1) * (t ** 2).sum(axis=1)) + 1e-12
    return num / den


def rank_models(rows: list, lead_times=LEAD_TIMES) -> str:
    """Rank methods by skill score relative to persistence (ref).
    see: https://en.wikipedia.org/wiki/Forecast_skill
    """

    metrics = {(method, k): (rmse, acc) for method, k, rmse, acc in rows}
    non_baseline = ["model", "ridge"]

    scores = {}
    for m in non_baseline:
        lead_scores = []
        for k in lead_times:
            rmse_m, acc_m = metrics[(m, k)]
            rmse_pers, _ = metrics[("persistence", k)]
            skill_rmse = 1.0 - rmse_m / max(rmse_pers, 1e-12)
            lead_scores.append(0.5 * skill_rmse + 0.5 * acc_m)
        scores[m] = float(np.mean(lead_scores))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best   = ranked[0][0]

    print(f"\n{'--- Model ranking (skill vs persistence) ---':}")
    print(f"{'method':<14}{'score':>10}")
    for m, s in ranked:
        marker = " best model" if m == best else ""
        print(f"{m:<14}{s:>10.4f}{marker}")

    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=config.DL_HORIZON)
    p.add_argument("--batchsize", type=int, default=1)
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
    ocean = ~test_ds.land_mask   # (H, W) bool

    # Fit Ridge on training split (needs n_out >= max lead time)
    print("Fitting Ridge baseline on training data…")
    train_ds = SSTADataset(
        config.SSTA_DAILY_PATH, config.LANDMASK_PATH, config.DL_TRAIN_RANGE,
        n_in=n_in, n_out=max(LEAD_TIMES),
    )
    ridge = RidgeBaseline(lead_times=LEAD_TIMES)
    ridge.fit(train_ds, test_ds.land_mask)

    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=2)
    ocean_t = torch.from_numpy(ocean).to(device)

    methods = ["model", "persistence", "ridge"]
    accum = {
        m: {k: {"sse": 0.0, "n": 0, "acc_sum": 0.0, "acc_n": 0}
            for k in LEAD_TIMES}
        for m in methods
    }

    for x, y in tqdm(loader, desc="evaluating"):
        x = x.to(device)
        y = y.to(device)

        preds = {
            "model":       autoregressive_rollout(model, x, args.horizon, n_out=n_out),
            "persistence": persistence_forecast(x, args.horizon),
            "ridge":       ridge.predict_all_leads(x.cpu(), args.horizon).to(device),
        }
        for m, pred in preds.items():
            for k in LEAD_TIMES:
                if k > pred.shape[1]:
                    continue
                pred_k = pred[:, k - 1][:, ocean_t]
                target_k = y[:, k - 1][:, ocean_t]
                a = accum[m][k]
                # RMSE
                a["sse"] += float(((pred_k - target_k) ** 2).sum().item())
                a["n"]   += pred_k.numel()

                # Spatial ACC: per-timestep Pearson
                p_np = pred_k.cpu().float().numpy()    # (B, n_ocean)
                t_np = target_k.cpu().float().numpy()
                accs = spatial_acc_batch(p_np, t_np)   # (B,)
                a["acc_sum"] += float(accs.sum())
                a["acc_n"]   += len(accs)

    rows = []
    print(f"\n{'method':<14}{'lead':>6}{'RMSE':>10}{'ACC':>10}")
    for m in methods:
        for k in LEAD_TIMES:
            a = accum[m][k]
            rmse = (a["sse"] / max(a["n"], 1)) ** 0.5
            acc = a["acc_sum"] / max(a["acc_n"], 1)
            rows.append((m, k, rmse, acc))
            print(f"{m:<14}{k:>6}{rmse:>10.4f}{acc:>10.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("method,lead_days,rmse,acc\n")
        for m, k, r, a in rows:
            f.write(f"{m},{k},{r:.6f},{a:.6f}\n")
    print(f"\nResults saved {args.out}")
    best = rank_models(rows)




if __name__ == "__main__":
    main()
