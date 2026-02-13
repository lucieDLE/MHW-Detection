import xarray as xr
import numpy as np

def load_dataset(path):
    return xr.open_dataset(path, engine="netcdf4")

def select_region(da, bounds):
    return da.sel(
        lat=slice(bounds["lat_max"], bounds["lat_min"]),
        lon=slice(bounds["lon_min"], bounds["lon_max"])
    )


def remove_countries_data(ds, ds_ref):
    indices = np.argwhere(np.isnan(ds_ref.sst[0].data))
    for (idx_y, idx_x) in indices:
        ds.sst[dict(lat=-idx_y, lon=idx_x)] = np.nan
    
    return ds
