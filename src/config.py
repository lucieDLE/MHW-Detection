DATA_PATH = "../data/sst.wkmean.1990-present.nc"
DATA_REF = "../data/sst.oisst.mon.ltm.1991-2020.nc"


MED_BOUNDS = {
    "lat_min": 20.5,
    "lat_max": 50.5,
    "lon_min": 0.5,
    "lon_max": 40.5
}

# BASELINE_PERIOD = (1990, 2020)
ROLLING_YEARS = 3
FREQ_PER_YEAR_MIN = 24
# EXTREME_QUANTILE = 0.95

WIDTH_PLOT=500
HEIGHT_RIGHT_PLOT = int(WIDTH_PLOT / 3)