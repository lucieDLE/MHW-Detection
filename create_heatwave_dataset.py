import xarray as xr

import os 
import pandas as pd
import numpy as np


def mark_runs_1d(arr, min_len):
    arr = np.asarray(arr, dtype=bool)
    n = arr.size
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    diff = np.diff(arr.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        ends = np.r_[ends, n]
    for s, e in zip(starts, ends):
        if (e - s) >= min_len:
            out[s:e] = True
    return out

def count_events_1d(arr, min_len=5):
    arr = np.asarray(arr, dtype=bool)
    if arr.size == 0:
        return 0
    diff = np.diff(arr.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        ends = np.r_[ends, arr.size]
    return np.sum((ends - starts) >= min_len)



MIN_DURATION = 5

ds = xr.open_zarr('data/sst_daily.zarr',
                    chunks={"time": 365, "lat": 180,"lon": 360}
                )

ds = ds.coarsen(lat=4, lon=4).mean()

da = ds.sst
da = da.resample(time="1D").mean()

# Compute threshold using a +/-5 day window across all years
rolling_window = 11  # 5 days before + current + 5 days after
window_radius = rolling_window // 2

da = da.assign_coords(
    year=da['time'].dt.year,
    dayofyear=da['time'].dt.dayofyear,
)

# Reshape to (dayofyear, year, lat, lon)
sst_yd = (
    da
    .set_index(time=['year', 'dayofyear'])
    .drop_duplicates('time')
    .unstack('time')
    .transpose('dayofyear', 'year', ...)
)

# Circular pad on dayofyear to support window at year boundaries
pad_left = sst_yd.isel(dayofyear=slice(-window_radius, None))
pad_right = sst_yd.isel(dayofyear=slice(0, window_radius))
sst_pad = xr.concat([pad_left, sst_yd, pad_right], dim='dayofyear')

# Rolling window over dayofyear, then compute quantile across (year, window)
threshold = (
    sst_pad
    .rolling(dayofyear=rolling_window, center=True)
    .construct('window')
    .quantile(0.9, dim=('year', 'window'))
)

# Keep the real dayofyear range (1..366)
threshold = threshold.isel(dayofyear=slice(window_radius, window_radius + sst_yd.sizes['dayofyear']))


# Align threshold (dayofyear, lat, lon) with full time series
thresh_time = threshold.sel(dayofyear=da['time'].dt.dayofyear)

# Boolean exceedance array over time
exceed = (da > thresh_time)
exceed = exceed.chunk({'time':-1})

mhw_mask = xr.apply_ufunc(
    mark_runs_1d,
    exceed,
    kwargs={"min_len": MIN_DURATION},
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[bool],
    dask_gufunc_kwargs={'allow_rechunk':True}
)



mhw_mask_ds = (
    mhw_mask
    .chunk({"time": -1, "lat": 90, "lon": 90})
)

mhw_mask_ds.to_zarr(
    "data/cache/mhw_mask.zarr",
    mode="w",
    consolidated=True,
)

mhw_mask = xr.open_zarr('data/cache/mhw_mask.zarr',
                        # chunks={"time": -1, "lat": 180,"lon": 360}
                        )

land_mask = da.isnull().all('time')

mhw_days_per_year = mhw_mask.groupby("time.year").sum("time")
mhw_days_per_year = mhw_days_per_year.where(~land_mask)
mhw_days_per_year_ds = mhw_days_per_year.compute()
# mhw_days_per_year_ds.to_zarr(
#     "data/cache/mhw_days_per_year.zarr",
#     mode="w",
#     consolidated=True,
# )


mhw_events_per_year = xr.apply_ufunc(
    count_events_1d,
    mhw_mask.groupby("time.year"),
    input_core_dims=[["time"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[int],
)

mhw_events_per_year = mhw_events_per_year.transpose('year', 'lat', 'lon')
mhw_events_per_year = mhw_events_per_year.where(~land_mask)

mhw_events_per_year_ds = mhw_events_per_year.compute()
# mhw_events_per_year_ds.to_zarr(
#     "data/cache/mhw_events_per_year.zarr",
#     mode="w",
#     consolidated=True,
# )


mhw_ds = mhw_days_per_year_ds.assign(
    day_per_year=(("year", "lat", "lon"), mhw_days_per_year_ds.sst.data),
    event_per_year=(("year", "lat", "lon"), mhw_events_per_year_ds.sst.data),
)

mhw_ds = mhw_ds.drop_vars(['sst','quantile'])

mhw_ds.to_zarr(
    "data/cache/mhw_per_year.zarr",
    mode="w",
    consolidated=True,
)
