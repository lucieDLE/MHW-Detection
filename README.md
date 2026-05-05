# MHW-Detection

## Overview

An interactive dashboard for exploring **Sea Surface Temperature (SST)** anomalies and **Marine Heatwave (MHW)** events from global ocean data spanning 1881–2026.

**Sea Surface Temperature (SST)** is the temperature of the uppermost layer of the ocean (top 1 mm to 10 m). Tracking SST anomalies — deviations from a long-term climatological baseline — reveals warming trends, regional variability, and the growing frequency of extreme ocean heat events.

**Marine Heatwaves (MHW)** are prolonged periods of anomalously warm ocean temperatures. They are formally defined (Hobday et al., 2016) as ≥5 consecutive days where SST exceeds the local 90th-percentile climatological threshold. MHWs have significant ecological and economic impacts, affecting coral reefs, fisheries, and coastal ecosystems.

The dashboard enables intuitive spatial-to-temporal climate exploration: users can scan global SST anomaly maps through time, detect long-term warming trends and extreme events at any location, and visualise MHW frequency and intensity year by year.

### Tools

| Layer | Libraries |
|---|---|
| Data & computation | `xarray`, `numpy`, `pandas`, `dask`, `scipy` |
| Storage | `zarr`, `netCDF4` |
| Visualisation | `hvPlot`, `HoloViews`, `Bokeh` |
| Dashboard | `Panel` |
| ML/DL | `scikit-learn` , `pytorch` |

## Dataset
The project uses the NOAA Optimum Interpolation (OI) SST V2 High Resolution [dataset](https://www.psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html). The dataset covers Sea Surface Temperature from 1881 to 2026 with weekly and biweekly resolution.

You will need to [download](https://www.psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html) the following datasets and place them in the parent directory inside a `data` folder:
- sst.week.mean.nc
- all daily data named as followed: sst.day.mean.{year}.nc


## Requirements
The app uses a conda environment where all the librairies are installed. To create the environment, run the following commands:
```
cd MHW-Detection
conda env create -f environment.yml
conda activate mhw-detection
```

## Running the Dashboard

#### 1. Preprocess (done only once)
Once the installation done you will first run the `preprocess.py` to create all the files needed to run the Dashboard. 
```
python preprocess.py
```

#### 2. Launch the Dashboard

```
panel serve app/interactive_map_panel.py --show
```

## Features

The dashboard has three tabs, each providing a different view of SST data.

### Tab 1 — SST Anomalies (Time Slider)

A video of the changes of SST anomalies in the world across time. The diverging colormap (blue = cooler, red = warmer) is centered on zero and clipped to ±5 °C, making it easy to scan regional warming and cooling events through time.

<img src="assets/images/dashboard_timeserie.png">

### Tab 2 — Anomaly Explorer

A spatial-to-temporal click-to-inspect workflow:

- **Variability Map** (left): displays the mean monthly standard deviation of SST anomalies across years, highlighting where the ocean fluctuates most. Clicking any location triggers the right panel.

  > **Note:** this map shows interannual variability, not heatwave intensity.

- **Interactive analyses** (right): clicking a grid cell generates three stacked plots:
  - **OLS trend**: long-term linear trend in °C/decade estimated by Ordinary Least Squares regression.
  - **Extreme events**: time series of SST anomalies with points above the 95th-percentile threshold highlighted in red.
  - **Event count histogram + KDE**: number of extreme events per year with a kernel-density curve to reveal multi-decadal shifts in frequency.

Below an example of the Anomaly Explorer Tab

<img src="assets/images/dashboard_anomaly.png">

### Tab 3 — Marine HeatWave Visualization

 Marine Heatwave (MHW) detection of 5 or more consecutive days where SST exceeds the local 90th-percentile climatological threshold.

- **Metric selector**: switch between *days per year* and *events per year*.
- **Year slider**: pan through annual MHW maps to inspect spatial patterns.
- **Click-to-inspect**: clicking the map plots a bar chart + KDE of the selected metric at that location across all available years.

Below an example of the Marine HeatWave Tab

<img src="assets/images/dashboard_mhw.png">



### Tab 4 — SST Forecasting

> **Note:** This tab will display data only after following the steps described in the [SST Forecasting](#sst-forecasting) section below.

This panel let you explore model performance spatially and at individual locations.

- **Metric selector**: choose from six maps based on the following metrics:
  - ACC: spatial Pearson correlation between predicted and observed SSTA per timestep, averaged over the test period. Range −1 to 1; higher is better.
  - RMSE: per-pixel Root Mean Square Error in °C; lower is better.
  - Forecasting Skill: improvement over persistence defined as `1 − RMSE_model / RMSE_persistence`. 
- **Lead time slider**: step through lead times from 1 to 28 days.

**Spatial metric maps**: a world map of the selected metric at a chosen lead time.

**Forecast trajectory chart**: clicking any point on the map draws a time series showing the observed input window, the model forecast, the ground truth, and the persistence baseline from a chosen start date. Use the **Forecast start date slider** to pick the anchor date.

Below an example of the SST Forecasting Tab

<img src="assets/images/dashboard_forecast.png">

## SST Forecasting

A deep-learning forecasting pipeline (`forecast/`) predicts daily SST anomalies up to 28 days ahead (called `lead times`). The pipeline is trained on 1982–2014 data, validated on 2015–2019, and evaluated on 2020–2025.
You can visit the [Forecast page](forecast/ANALYSIS.md) for more details of the different models, parameters, lead time and performances.

### Training a Model

```bash
python forecast/train.py --model conv_lstm --epochs 30 --n_in 14 --batchsize 2

# resuming a training
python forecast/train.py --model conv_lstm --epochs 30 --n_in 14 --batchsize 2 --resume checkpoint.ckpt

# monitor training
tensorboard --logdir forecast/runs
```

Checkpoints are saved to `forecast/checkpoints/`.
> **Note:** All models are intentionally lightweight — trained for fewer than 100 epochs with small batch sizes due to hardware constraints. More expressive architectures and longer training runs would likely yield significant performance gains.


### Evaluating Performance

```bash
python forecast/evaluate.py --checkpoint forecast/checkpoints/conv_lstm_best.pt --horizon 28
```

Outputs a CSV (`eval_results.csv`) with RMSE and ACC for each method × lead time, ranks models by composite skill score relative to the persistence baseline.
- **RMSE** (Root Mean Square Error): Measures the average magnitude of forecast errors, in the same unit as the anomaly (°C). Lower is better.
- **ACC** (Anomaly Correlation Coefficient): Measures the spatial or temporal correlation between predicted and observed anomalies. Ranges from −1 to 1, where 1 indicates a perfect forecast and values above 0.6 are generally considered skillful in climate forecasting.
- Forecasting **Skill Score**: Measures improvement over the persistence baseline. A score of 0 means the model performs no better than persistence; a score of 1 means perfect forecasting; a negative score means the model performs worse than persistence.

### Export Spatial Maps
After training, you can export per-pixel predicted STTA, RMSE, and ACC maps for visualization:
 
```bash
# --metrics will export the RMSE and ACC
# --forecast will export the predicted SSTA
python forecast/export_to_dataset.py --checkpoint forecast/checkpoints/conv_lstm_best.pt --forecast 1 --metrics 1 --out data/cache
```

This generates `{model_name}_{n_in}_ACC_RMSE.zarr` and `{model_name}_{n_in}_forecast.zarr`, containing per-pixel ACC and RMSE world maps for every lead time, for both the trained model and the persistence baseline. Once the output paths are set via the `FORECAST_ACC_PATH` and `FORECAST_CHART_PATH` variables in the config file, it is loaded automatically by the SST ForecastinFORECAST_DIFF_ACC_CAPTIONg tab of the dashboard.


## References

- Reynolds, R.W., N.A. Rayner, T.M. Smith, D.C. Stokes, and W. Wang, 2002: An improved in situ and satellite SST analysis for climate. J. Climate, 15, 1609-1625.
- Hobday, Alistair J., et al. "A hierarchical approach to defining marine heatwaves." Progress in oceanography 141 (2016): 227-238.