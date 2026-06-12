import xarray as xr
import config

## coarsen dat high res for demo on HF

ds = xr.open_zarr(config.ANOMALY_MAP_PATH)              # 720×1440
ds_c = ds.coarsen(lat=4, lon=4, boundary="trim").mean() # → 180×360, ~330 MB

ds_c = ds_c.chunk({"time": 365, "lat": 180, "lon": 360})
# drop stale per-variable chunk encodings inherited from the source zarr
for v in ds_c.variables:
    ds_c[v].encoding.clear()
    
ds_c.to_zarr("data/cache_deploy/ssta_high_res.zarr", mode="w")
