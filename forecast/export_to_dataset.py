import argparse
import sys
from pathlib import Path

import torch.nn as nn
from tqdm import tqdm
import hvplot
import hvplot.xarray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast.evaluate import reduce_factor, spatial_acc_batch
from forecast.rollout import autoregressive_rollout
from forecast.baselines import persistence_forecast, RidgeBaseline
from forecast.data import *
from forecast.model import *
from config import *


def pixel_acc_map(pred: np.ndarray, truth: np.ndarray, axis:int) -> np.ndarray:
    # pred, truth: (N, H, W)
    p = pred  - pred.mean(axis=axis, keepdims=True)
    t = truth - truth.mean(axis=axis, keepdims=True)
    num = (p * t).sum(axis=axis)
    den = np.sqrt((p**2).sum(axis=axis) * (t**2).sum(axis=axis)) + 1e-12
    return num / den   # (H, W)

def pixel_rmse_map(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
     # pred, truth: (N, H, W)
     return np.sqrt(np.mean((pred - truth) ** 2, axis=0))  # (H, W)



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="best model", required=True, type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "forecast" / "eval_results.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt["model"]
    n_in, n_out = ckpt["n_in"], ckpt["n_out"]
    model = build_model(model_name, n_in=n_in, n_out=n_out)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()


    print(f"Loaded {args.checkpoint.name}  n_in={n_in}  n_out={n_out}")

    ds_xr = xr.open_zarr(str(config.SSTA_DAILY_PATH))
    horizon = max(config.LEAD_TIMES)

    test_ds = SSTADataset(
        config.SSTA_DAILY_PATH,
        config.LANDMASK_PATH,
        config.DL_TEST_RANGE,
        n_in=n_in,
        n_out=horizon, # the number of prediction days we want, not the days the model was trained with
    )






    ocean = ~test_ds.land_mask   # (H, W) bool

    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    ocean_t = torch.from_numpy(ocean).to(device)

    methods = ["model", "persistence"]
    accum = {
        m: {k: {"sse": 0.0, "n": 0, "acc_sum": 0.0, "acc_n": 0}
            for k in config.LEAD_TIMES}
        for m in methods
    }

    spatial_preds_model  = {k: [] for k in config.LEAD_TIMES}
    spatial_preds_persistence  = {k: [] for k in config.LEAD_TIMES}
    spatial_truths = {k: [] for k in config.LEAD_TIMES}

    for x, y in tqdm(loader, desc="evaluating"):
        preds_model = autoregressive_rollout(model, x, horizon, n_out=n_out)
        preds_persistence = persistence_forecast(x, horizon)

        for k in config.LEAD_TIMES:
            if k <= preds_model.shape[1]:            
                pred_m_k = preds_model[:, k - 1]
                pred_p_k = preds_persistence[:, k - 1]
                target_k = y[:, k - 1]

                # set all land at 0 insteak of masking to preserve shape
                pred_m_k[:, test_ds.land_mask] = 0
                pred_p_k[:, test_ds.land_mask] = 0
                target_k[:, test_ds.land_mask] = 0

                spatial_preds_model[k].append(pred_m_k)   # (B, H, W)
                spatial_preds_persistence[k].append(pred_p_k)
                spatial_truths[k].append(target_k)


    acc_model_worldmap = []
    acc_persistence_worldmap = []
    rmse_model_worldmap = []
    rmse_persistence_worldmap = []

    for k in config.LEAD_TIMES:
        a = accum[m][k]

        all_preds_model  = np.concatenate(spatial_preds_model[k],  axis=0)
        all_preds_persistence  = np.concatenate(spatial_preds_persistence[k],  axis=0)
        all_truth = np.concatenate(spatial_truths[k], axis=0)
        
        # ACC
        acc_model = pixel_acc_map(all_preds_model, all_truth, axis=0)
        acc_model[test_ds.land_mask] = np.nan
        acc_model_worldmap.append(acc_model)

        acc_persistence = pixel_acc_map(all_preds_persistence, all_truth, axis=0)
        acc_persistence[test_ds.land_mask] = np.nan
        acc_persistence_worldmap.append(acc_persistence)

        # RMSE
        rmse_model = pixel_rmse_map(all_preds_model, all_truth)
        rmse_model[test_ds.land_mask] = np.nan
        rmse_model_worldmap.append(rmse_model)

        rmse_persistence = pixel_rmse_map(all_preds_persistence, all_truth)
        rmse_persistence[test_ds.land_mask] = np.nan
        rmse_persistence_worldmap.append(rmse_persistence)

    acc_model_worldmap = np.array(acc_model_worldmap)
    acc_persistence_worldmap = np.array(acc_persistence_worldmap)

    rmse_model_worldmap = np.array(rmse_model_worldmap)
    rmse_persistence_worldmap = np.array(rmse_persistence_worldmap)
    rmse_max = max(np.nanquantile(rmse_model_worldmap, 0.99), np.nanquantile(rmse_persistence_worldmap, 0.99))

    ds_xr = ds_xr.coarsen(lat=2, lon=2).mean()
    ds_xr.ssta.shape

    ds = xr.Dataset(
        data_vars=dict(
            model_acc=(['lead_time', 'lat', 'lon'], acc_model_worldmap),
            persistence_acc=(['lead_time', 'lat', 'lon'], acc_persistence_worldmap),
            model_rmse=(['lead_time', 'lat', 'lon'], rmse_model_worldmap),
            persistence_rmse=(['lead_time', 'lat', 'lon'], rmse_persistence_worldmap),
            rmse_range = np.array([0,rmse_max]),

        ),
        coords = dict(
            lead_time=np.array(config.LEAD_TIMES),
            lat=("lat", ds_xr.ssta.lat.values),
            lon=("lon", ds_xr.ssta.lon.values),
        ),
        attrs=dict(
            metric='ACC_RMSE',
            acc_metric='Anomaly Correlation Coefficient',
            rmse_metric='Root Mean Square Error',
            model=model_name,
            input_window=n_in)
    )

    ds.to_zarr(f'{ds.model}_n_in{ds.input_window}_{ds.metric}.zarr')


if __name__ == "__main__":
    main()
