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
MHW_SLIDER_WIDTH = 200

# Cache outputs
INITIAL_MAP_CACHE = "data/cache/initial_map.zarr"
ANOMALY_MAP_PATH = 'data/cache/ssta_high_res.zarr'
MHW_MAP_PATH = 'data/cache/mhw.zarr'
SSTA_DAILY_PATH = "data/cache/ssta_daily.zarr"        # daily SST anomaly
CLIM_PATH       = "data/cache/clim_daily.zarr"        # day-of-year climatology
LANDMASK_PATH   = "data/cache/landmask_daily.zarr"    # land mask (180×180)
# Video output
SSTA_VIDEO_PATH = "assets/videos/sst_weekly_combined.mp4"

# Chronological splits — strict, no shuffling
DL_TRAIN_RANGE = ("1982-01-01", "2014-12-31")
DL_VAL_RANGE   = ("2015-01-01", "2019-12-31")
DL_TEST_RANGE  = ("2020-01-01", "2025-12-31")

# Model / training hyperparameters
DL_N_IN          = 14         # input window length (days)
DL_N_OUT         = 1          # one-step model; multi-day done via rollout
DL_HORIZON       = 14         # rollout  for evaluation
DL_COARSEN_FACTOR = 2         # coarsen the map to avoid OOM
BATCH_SIZE    = 8
LR            = 1e-3
EPOCHS        = 30
NUM_WORKERS   = 4
CHECKPOINT_DIR = "forecast/checkpoints"
LOG_DIR        = "forecast/runs"
