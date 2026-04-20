from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

import config


class SSTADataset(Dataset):
    """Sliding-window dataset over the daily SSTA 

    Each sample is (x, y) where:
        x  shape (n_in, H, W)   — SSTA at days [t-n+1, …, t]
        y  shape (n_out, H, W)  — SSTA at days [t+1, …, t+n]

    Land cells are filled with 0; the land mask is exposed via .land_mask
    so the training loop can apply it to the loss.
    """

    def __init__(
        self,
        ssta_path: str | Path,
        landmask_path: str | Path,
        time_range: Tuple[str, str],
        n_in: int = 14,
        n_out: int = 1,
    ):
        self.n_in = n_in
        self.n_out = n_out

        ds = xr.open_zarr(str(ssta_path))
        ssta = ds.sst.sel(time=slice(*time_range))

        self.ssta = ssta.load().astype("float32").values
        self.times = ssta.time.values

        land_mask = xr.open_zarr(str(landmask_path)).land_mask.values.astype(bool)
        self.land_mask = land_mask  # (H, W); True over land

        np.nan_to_num(self.ssta, copy=False, nan=0.0) # convert the land set to nan to 0 instead

        T = self.ssta.shape[0] # number of time points
        self.valid_starts = np.arange(0, T - n_in - n_out + 1) # remove the input/output window length

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        t0 = self.valid_starts[idx]
        x = self.ssta[t0 : t0 + self.n_in]
        y = self.ssta[t0 + self.n_in : t0 + self.n_in + self.n_out]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(
    ssta_path: str | Path = config.DL_SSTA_DAILY_PATH,
    landmask_path: str | Path = config.DL_LANDMASK_PATH,
    n_in: int = config.DL_N_IN,
    n_out: int = config.DL_N_OUT,
    batch_size: int = config.DL_BATCH_SIZE,
    num_workers: int = config.DL_NUM_WORKERS,
):
    train_ds = SSTADataset(ssta_path, landmask_path, config.DL_TRAIN_RANGE, n_in, n_out)
    val_ds   = SSTADataset(ssta_path, landmask_path, config.DL_VAL_RANGE,   n_in, n_out)
    test_ds  = SSTADataset(ssta_path, landmask_path, config.DL_TEST_RANGE,  n_in, n_out)

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader, train_ds.land_mask
