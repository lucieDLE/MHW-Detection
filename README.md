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
| ML | `scikit-learn` |

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

<img src="assets/images/dashboard_ssta_analysis.png">

### Tab 3 — Marine HeatWave Visualization

 Marine Heatwave (MHW) detection of 5 or more consecutive days where SST exceeds the local 90th-percentile climatological threshold.

- **Metric selector**: switch between *days per year* and *events per year*.
- **Year slider**: pan through annual MHW maps to inspect spatial patterns.
- **Click-to-inspect**: clicking the map plots a bar chart + KDE of the selected metric at that location across all available years.

Below an example of the Marine HeatWave Tab

<img src="assets/images/mhw_analysis.png">


## Training a model

```
(some code)

# to visualize the training 
tensorboard --logdir forecast/runs
```



## References

- Reynolds, R.W., N.A. Rayner, T.M. Smith, D.C. Stokes, and W. Wang, 2002: An improved in situ and satellite SST analysis for climate. J. Climate, 15, 1609-1625.
- Hobday, Alistair J., et al. "A hierarchical approach to defining marine heatwaves." Progress in oceanography 141 (2016): 227-238.