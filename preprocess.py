#!/usr/bin/env python
"""
preprocess.py — Run this once before launching the SST Dashboard.

Produces every cache file the app reads at startup:

    data/cache/ssta_high_res.zarr   Weekly SST anomalies  (time-slider + anomaly tabs)
    data/cache/initial_map.zarr     SST variability map   (anomaly tab background)
    data/cache/mhw_threshold.zarr   Climatological threshold (intermediate, reusable)
    data/cache/mhw.zarr             MHW days + events/year  (MHW tab)

Usage
-----
    python preprocess.py                  # run all steps
    python preprocess.py --skip-weekly    # skip SSTA + initial-map
    python preprocess.py --skip-mhw       # skip MHW pipeline
    python preprocess.py --force          # overwrite existing caches
    python preprocess.py --coarsen 2      # spatial coarsening factor for daily data (default 4)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import dask.array as dsa
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from scipy.ndimage import label as scipy_label

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

# DAILY_SST_FOLD  = ROOT / "data" / "daily"
DAILY_SST_PATH = ROOT / "data" / "sst_daily.zarr"
WEEKLY_SST_PATH = ROOT / config.DATA_PATH

SSTA_PATH      = ROOT / config.ANOMALY_MAP_PATH
INIT_MAP_PATH  = ROOT / config.INITIAL_MAP_CACHE
THRESHOLD_PATH = ROOT / "data" / "cache" / "mhw_threshold.zarr"
MHW_MASK_PATH  = ROOT / "data" / "cache" / "mhw_mask.zarr"
MHW_PATH       = ROOT / config.MHW_MAP_PATH


# ── Step 1 — Weekly SSTA ─────────────────────────────────────────────────────

def build_ssta(force: bool) -> None:
    if SSTA_PATH.exists() and not force:
        print(f"\n {SSTA_PATH} already exists \n")
        return

    print(f"\n   Loading {WEEKLY_SST_PATH} … \n")
    ds = xr.open_dataset(WEEKLY_SST_PATH, engine="netcdf4", chunks=config.CHUNKS)

    sst_gb   = ds.sst.groupby("time.month")
    tos_clim = sst_gb.mean(dim="time")
    ssta     = (sst_gb - tos_clim).astype("float32")

    SSTA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n   Writing {SSTA_PATH} … \n")
    ssta.to_dataset(name="sst").to_zarr(str(SSTA_PATH), mode="w", consolidated=True)
    print(f"\n   ✓ SSTA saved \n")


# ── Step 2 — Initial variability map ─────────────────────────────────────────

def build_initial_map(force: bool) -> None:
    if INIT_MAP_PATH.exists() and not force:
        print(f"\n {INIT_MAP_PATH} already exists \n")
        return

    print(f"\n   Loading {WEEKLY_SST_PATH} … \n")
    ds = xr.open_dataset(WEEKLY_SST_PATH, engine="netcdf4", chunks=config.CHUNKS)

    sst = ds.sst.sel(time=slice(config.MIN_DATE, config.MAX_DATE))
    if config.MAP_COARSEN and config.MAP_COARSEN > 1:
        sst = sst[:: config.TIME_COARSEN, :: config.MAP_COARSEN, :: config.MAP_COARSEN]

    sst_grouped = sst.groupby("time.month")
    tos_std     = sst_grouped.std(dim="time")
    initial_map = tos_std.mean(dim="month").astype("float32").compute()

    INIT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n   Writing {INIT_MAP_PATH} … \n")
    initial_map.to_zarr(str(INIT_MAP_PATH), mode="w", consolidated=True)
    print(f"\n   ✓ Initial map saved \n")


# ── Step 3a — Build daily zarr from individual NetCDF files ──────────────────

def build_initial_daily_file(force: bool, coarsen_factor: int) -> None:
    if DAILY_SST_PATH.exists() and not force:
        print(f"\n {DAILY_SST_PATH} already exists \n")
        return

    all_nc = sorted(glob.glob(os.path.join(daily_files_path, "*.nc")))
    all_nc = [f for f in all_nc if "1986" not in f]  # 1986 file is corrupt
    print(f"Found {len(all_nc)} files")

    ds_all = xr.open_mfdataset(
        all_nc,
        combine="by_coords",
        chunks={"time": 365, "lat": 360, "lon": 360},
    )
    ds_all = ds_all.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary="trim").mean()
    ds_all = ds_all.chunk({"time": 365, "lat": 360, "lon": 360})  # rechunk after coarsen

    DAILY_SST_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n Writing {DAILY_SST_PATH} … \n")
    ds_all.to_zarr(str(DAILY_SST_PATH), mode="w", consolidated=True)
    print(f"\n Daily SST saved \n")


# ── Step 3b — Climatological threshold ───────────────────────────────────────

def _threshold_block(block, doy, window_radius, q):
    T, H, W = block.shape
    thresh = np.full((366, H, W), np.nan, dtype=np.float64)
    for d in range(1, 367):
        lo   = (d - 1 - window_radius) % 366 + 1
        hi   = (d - 1 + window_radius) % 366 + 1
        mask = (doy >= lo) & (doy <= hi) if lo <= hi else (doy >= lo) | (doy <= hi)
        if mask.sum() > 0:
            thresh[d - 1] = np.nanpercentile(block[mask], q * 100, axis=0)
    return thresh


def build_threshold(da: xr.DataArray, tile: int, force: bool) -> xr.DataArray:
    if THRESHOLD_PATH.exists() and not force:
        print(f"\n threshold cache exists, reloading … \n")
        return xr.open_zarr(str(THRESHOLD_PATH)).threshold

    doy_values = da.time.dt.dayofyear.values

    da_tiled = da.chunk({"time": -1, "lat": tile, "lon": tile})

    threshold_dask = dsa.map_blocks(
        _threshold_block,
        da_tiled.data,
        drop_axis=0,
        new_axis=0,
        chunks=((366,), da_tiled.data.chunks[1], da_tiled.data.chunks[2]),
        dtype=np.float64,
        doy=doy_values,
        window_radius=5,
        q=0.9,
    )

    threshold = xr.DataArray(
        threshold_dask,
        dims=["dayofyear", "lat", "lon"],
        coords={"dayofyear": np.arange(1, 367), "lat": da.lat, "lon": da.lon},
        name="threshold",
    )

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n   Writing {THRESHOLD_PATH} … \n")
    threshold.to_dataset(name="threshold").to_zarr(str(THRESHOLD_PATH), mode="w", consolidated=True)
    print(f"\n   ✓ Threshold saved \n")
    return xr.open_zarr(str(THRESHOLD_PATH)).threshold


# ── 3b. MHW mask ────────────────────────────────────────────────────────────

def _make_mhw_block_fn():
    """Return the fastest available _mhw_block implementation."""
    try:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _mark_runs_numba(flat, min_len):
            N, T = flat.shape
            out = np.zeros((N, T), dtype=np.bool_)
            for i in prange(N):
                j = 0
                while j < T:
                    if flat[i, j]:
                        k = j + 1
                        while k < T and flat[i, k]:
                            k += 1
                        if k - j >= min_len:
                            out[i, j:k] = True
                        j = k
                    else:
                        j += 1
            return out

        def _mhw_block(block, min_len):
            T, H, W = block.shape
            flat = np.asfortranarray(block.reshape(T, H * W).T)
            return _mark_runs_numba(flat, min_len).T.reshape(T, H, W)

        print("  Using numba MHW mask (parallel compiled)")
        return _mhw_block

    except ImportError:
        from scipy.ndimage import label as _label

        def _mhw_block(block, min_len):
            T, H, W = block.shape
            flat = block.reshape(T, H * W)
            out  = np.zeros_like(flat, dtype=bool)
            for i in range(flat.shape[1]):
                col = flat[:, i]
                labeled, n = _label(col)
                if n:
                    sizes = np.bincount(labeled)[1:]
                    valid = np.where(sizes >= min_len)[0] + 1
                    out[:, i] = np.isin(labeled, valid)
            return out.reshape(T, H, W)

        return _mhw_block


def build_mhw_mask(da: xr.DataArray, threshold: xr.DataArray,
                   tile: int, min_duration: int) -> xr.DataArray:

    thresh_time = threshold.sel(dayofyear=da.time.dt.dayofyear)
    exceed = (da > thresh_time).chunk({"time": -1, "lat": 30, "lon": 30})

    _mhw_block = _make_mhw_block_fn()

    mask_data = dsa.map_blocks(
        _mhw_block,
        exceed.data,
        dtype=bool,
        chunks=exceed.data.chunks,
        min_len=min_duration,
    )
    return xr.DataArray(mask_data, dims=exceed.dims, coords=exceed.coords, name="mhw_mask")


# ── 3c. Days + events per year ──────────────────────────────────────────────

def _make_count_events_fn():
    """Return the fastest available batched event-count implementation."""
    try:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _count_numba(flat, min_len):
            N, T = flat.shape
            out = np.zeros(N, dtype=np.int32)
            for i in prange(N):
                count = 0
                j = 0
                while j < T:
                    if flat[i, j]:
                        k = j + 1
                        while k < T and flat[i, k]:
                            k += 1
                        if k - j >= min_len:
                            count += 1
                        j = k
                    else:
                        j += 1
                out[i] = count
            return out

        def _count_events_batch(arr, min_len=5):
            # apply_ufunc passes (lat, lon, T) with time as core dim
            H, W, T = arr.shape
            flat = np.asfortranarray(arr.reshape(H * W, T))
            return _count_numba(flat, min_len).reshape(H, W)

        return _count_events_batch

    except ImportError:
        from scipy.ndimage import label as _label

        def _count_events_batch(arr, min_len=5):
            H, W, T = arr.shape
            flat = arr.reshape(H * W, T)
            out  = np.zeros(H * W, dtype=np.int32)
            for i in range(H * W):
                labeled, n = _label(flat[i])
                if n:
                    sizes = np.bincount(labeled)[1:]
                    out[i] = int(np.sum(sizes >= min_len))
            return out.reshape(H, W)

        return _count_events_batch

def build_mhw_dataset(force: bool, coarsen: int) -> None:
    if MHW_PATH.exists() and not force:
        print(f"\n {MHW_PATH} already exists \n")
        return

    if not DAILY_SST_PATH.exists():
        print(f"\n Daily SST not found at {DAILY_SST_PATH} \n")
        return

    min_duration = 5

    print(f"\n   Loading {DAILY_SST_PATH} … \n")
    ds = xr.open_zarr(str(DAILY_SST_PATH), chunks={"time": 365, "lat": 180, "lon": 360})
    da = ds.sst

    print(da.shape)

    _, unique_idx = np.unique(da.time.values, return_index=True)
    if len(unique_idx) < len(da.time):
        print(f"\n   Dropped {len(da.time) - len(unique_idx)} duplicate timestamps \n")
        da = da.isel(time=unique_idx)

    da = da.chunk({"time": -1, "lat": 30, "lon": 30})

    land_mask = da.isnull().all("time")

    print("  Step 3b — Climatological threshold …")
    t0 = time.time()
    threshold = build_threshold(da, tile, force)
    print(f"\n   done in {time.time() - t0:.0f}s \n")

    print("  Step 3c — MHW mask …")
    t0 = time.time()
    if MHW_MASK_PATH.exists() and not force:
        print(f"\n {MHW_MASK_PATH} already exists \n")
        mhw_mask = xr.open_zarr(str(THRESHOLD_PATH)).mhw_mask

    else:

        mhw_mask = (
            build_mhw_mask(da, threshold, tile, min_duration)
            .where(~land_mask)
            .transpose("time", "lat", "lon")
        )
        mhw_mask.to_dataset(name="mhw_mask").to_zarr(str(MHW_MASK_PATH), mode="w", consolidated=True)
        print(f"\n   done in {time.time() - t0:.0f}s \n")

    print("  Step 3d — MHW days per year …")
    mhw_days = (
        mhw_mask.groupby("time.year").sum("time")
        .where(~land_mask)
        .rename("day_per_year")
        .astype(np.float32)
    )

    # ── Events per year ───────────────────────────────────────────────────────
    print("  Step 3d — MHW events per year …")
    _count_events_batch = _make_count_events_fn()

    mhw_events = xr.apply_ufunc(
        _count_events_batch,
        mhw_mask.groupby("time.year"),
        input_core_dims=[["time"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.int32],
        kwargs={"min_len": min_duration},
    )
    mhw_events = (
        mhw_events
        .transpose("year", "lat", "lon")
        .where(~land_mask)
        .rename("event_per_year")
        .astype(np.float32)
    )

    print("  Computing and writing MHW dataset …")
    t0 = time.time()
    mhw_ds = xr.Dataset({
        "day_per_year": mhw_days, 
        "event_per_year": mhw_events
        })
    MHW_PATH.parent.mkdir(parents=True, exist_ok=True)
    mhw_ds.chunk({"year": 1, "lat": 90, "lon": 90}).to_zarr(str(MHW_PATH), mode="w", consolidated=True)
    print(f"\n   ✓ MHW dataset saved in {time.time() - t0:.0f}s \n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-weekly", action="store_true", help="Skip SSTA + initial-map")
    p.add_argument("--skip-mhw",   action="store_true", help="Skip MHW pipeline")
    p.add_argument("--force",      action="store_true", help="Overwrite existing caches")
    p.add_argument("--coarsen",    type=int, default=2, help="Spatial coarsening factor (default: 4)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SST Dashboard — preprocessing")
    print("=" * 60)

    if not args.skip_weekly:
        print("\n[Step 1] Weekly SST anomalies")
        build_ssta(args.force)

        print("\n[Step 2] Initial variability map")
        build_initial_map(args.force)
    else:
        print("\n[Step 1 + 2] Skipped (--skip-weekly)")

    if not args.skip_mhw:
        cluster = LocalCluster(n_workers=6, threads_per_worker=2, memory_limit="16GB")
        client  = Client(cluster)
        print(f"\n   You can follow the process with the following Dashboard: {client.dashboard_link} \n")

        print("\n[Step 3a] Daily SST zarr - can take up to 10mins")
        build_initial_daily_file(args.force, args.coarsen)


        print(f"\n \n[Step 3b–e] MHW pipeline  (coarsen={args.coarsen}×) \n")
        try:
            t0 = time.time()
            build_mhw_dataset(args.force, args.coarsen)
            print(f"\n   Total MHW elapsed: {time.time() - t0:.0f}s \n")
        finally:
            client.close()
            cluster.close()
    else:
        print("\n[Step 3] Skipped (--skip-mhw)")

    print("\n" + "=" * 60)
    print("Done. Run the app with:")
    print("  panel serve app/interactive_map_panel.py --show")
    print("=" * 60)


if __name__ == "__main__":
    main()
