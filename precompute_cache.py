from pathlib import Path

import xarray as xr

import config
from app.interactive_map_panel import load_initial_map, _resolve_data_path


def main():
    cache_path = _resolve_data_path(config.INITIAL_MAP_CACHE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    initial_map = load_initial_map()
    # Ensure it is computed and written once
    initial_map.astype("float32").to_netcdf(cache_path)
    print(f"Saved initial map cache to {cache_path}")


if __name__ == "__main__":
    main()
