
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
    """
    p = pred_ocean  - pred_ocean.mean( axis=1, keepdims=True)
    t = truth_ocean - truth_ocean.mean(axis=1, keepdims=True)
    num = (p * t).sum(axis=1)
    den = np.sqrt((p ** 2).sum(axis=1) * (t ** 2).sum(axis=1)) + 1e-12
    return num / den


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=config.DL_HORIZON)
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--mhw", help="compute MHW detection classification")
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

    if args.mhw:
        evaluate_mhw_detection(model, test_ds, ocean, args, device, n_in, n_out)


def evaluate_mhw_detection(model, test_ds, ocean, args, device, n_in, n_out):
    """compute MHW map from predicted SSTA and compare to true MHW mask.
    Reuses the existing threshold cache from preprocess.py.
    """
    threshold_path = ROOT / "data" / "cache" / "mhw_threshold.zarr"

    print("\nComputing MHW classification …")
    predicted_map_shape = test_ds[0][0].shape
    thresh_ds =reduce_factor(xr.open_zarr(threshold_path) , predicted_map_shape)
    clim_ds =reduce_factor(xr.open_zarr(str(config.CLIM_PATH)) , predicted_map_shape)
        
    thresh = thresh_ds.threshold.load().values   # original (366, 360, 720)
    clim = clim_ds.clim.load().values   # original (366, 180, 360)

    # Times in test_ds: prediction at index i corresponds to truth time test_ds.times[start+n_in+k-1]
    times = test_ds.times
    starts = test_ds.valid_starts

    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=2)
    ocean_t = torch.from_numpy(ocean).to(device)

    # aggregated TP/FP/FN
    stats = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in LEAD_TIMES}

    sample_i = 0
    for x, y in tqdm(loader):
        x = x.to(device)

        pred = autoregressive_rollout(model, x, args.horizon, n_out=n_out)
        pred = pred.detach().cpu().numpy()
        truth = y.numpy()

        for idx in range(args.batchsize):
            t0 = starts[sample_i + idx]
            for k in LEAD_TIMES:
                # t_target is the day we try to predict, meaning starting from a 14 days window we fed, we try to do the prediction at day k
                # hence the final day is t0 + 14 + k
                target_time = t0 + test_ds.n_in + k - 1
                # if target_time < len(times):
                t_target = times[t0 + test_ds.n_in + k - 1] 

                # since we trying to predict heatwave, we need to get the real temperature and not the anomalies.
                # from the ssta to get the sst we can just add the climatology, because ssta(t) = sst(t) - clim.
                # clim is of shape 366, the average temperature per dayofyear.

                # here we get the conversion of the day/month/year to the day number per year [0, 366]
                doy = int(np.datetime64(t_target, "D").astype(object).timetuple().tm_yday)
                
                # in addition of the climatology at doy we get the threshold value to see if the predicted
                # value is exceeding it or not. (--> is it a heatwave or not)
                clim_k   = clim[doy -1] # (H, W)
                thresh_k = thresh[doy -1] # (H, W)

                # convert ssta to sst 
                p_sst = pred[idx, k - 1] + clim_k
                t_sst = truth[idx, k - 1] + clim_k

                # worldmap with pixels 0 and 1 of exceed threshold or not
                p_mhw = (p_sst > thresh_k) & ocean
                t_mhw = (t_sst > thresh_k) & ocean

                # build confusion matrices for all predicted time k
                tp = int(np.sum(p_mhw & t_mhw))
                fp = int(np.sum(p_mhw & ~t_mhw & ocean))
                fn = int(np.sum(~p_mhw & t_mhw & ocean))
                tn = int(np.sum(~p_mhw & ~t_mhw & ocean))
                
                s = stats[k]
                s["tp"] += tp
                s["fp"] += fp
                s["fn"] += fn
                s["tn"] += tn
                
        sample_i += args.batchsize # we move to the next days to predict
        # break

    print(f"\n{'lead':>6}{'precision':>12}{'recall':>10}{'F1':>10}")
    mhw_csv = args.out.with_name("eval_mhw_skill.csv")
    with mhw_csv.open("w") as f:
        f.write("lead_days,precision,recall,f1,tp,fp,fn,tn\n")
        for k in LEAD_TIMES:
            s = stats[k]
            prec = s["tp"] / max(s["tp"] + s["fp"], 1)
            rec  = s["tp"] / max(s["tp"] + s["fn"], 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            print(f"{k:>6}{prec:>12.4f}{rec:>10.4f}{f1:>10.4f}")
            f.write(f"{k},{prec:.6f},{rec:.6f},{f1:.6f},{s['tp']},{s['fp']},{s['fn']},{s['tn']}\n")

            cnf_matrix = np.array([[s['tn'], s['fp']], [s['fn'], s['tp']]])
            
            fig = plt.figure(figsize=(9,4))
            plt.subplot(121)
            plot_confusion_matrix(cnf_matrix, classes=["MHW","No MHW"])
            plt.title(f'Confusion matrix at lead time={k}')

            plt.subplot(122)
            plot_confusion_matrix(cnf_matrix, classes=["MHW","No MHW"], normalize=True)
            plt.title(f'Normalized Confusion matrix')
            plt.tight_layout()
            fig.savefig(args.out.with_name(f'confusion_matrices_mhw_k{k}.png'))

    print(f"saved output: {mhw_csv}")


if __name__ == "__main__":
    main()
