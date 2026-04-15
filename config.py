# Data
DATA_PATH = "data/sst.week.mean.nc"


# Performance tuning for high-res data
CHUNKS = {"time": 48, "lat": 180, "lon": 180}
MAP_COARSEN = 2
RAW_COARSEN = 2
TIME_COARSEN = 3

MIN_YEAR = 1990
MAX_YEAR = 2022

MIN_DATE="1982-01-01"
MAX_DATE="2025-12-31"

ROLLING_YEARS = 3
FREQ_PER_YEAR_MIN = 52
EXTREME_QUANTILE = 0.95
MIN_DURATION = 5 # 5 days 


# Display
WIDTH_PLOT=500
HEIGHT_RIGHT_PLOT = int(WIDTH_PLOT / 3)

DEFAULT_TAP_LON = 300
DEFAULT_TAP_LAT = 40

MAP_WIDTH = 980
MAP_HEIGHT = 560
RIGHT_PANEL_WIDTH = 560
RIGHT_PLOT_HEIGHT = 335

TIME_SERIE_WIDTH = 1440
TIME_SERIE_HEIGHT = 760

# Cache outputs
INITIAL_MAP_CACHE = "data/cache/initial_map.zarr"
ANOMALY_MAP_PATH = 'data/cache/ssta_high_res.zarr'
MHW_MAP_PATH = 'data/cache/mhw.zarr'
# Video output
RAW_VIDEO_PATH = "assets/sst_weekly.mp4"
