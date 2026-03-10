# MHW-Detection

## Overview
Interactive exploration of **Sea Surface Temperature (SST)**  anomalies.

This project provides an interactive dashboard to explore long-term SST anomaly patterns using xarray, hvPlot, Holoviews, and Panel. Users can click anywhere on a spatial SST anomaly map and instantly generate detailed time-series analyses for that location.
The goal is to enable intuitive spatial-to-temporal climate exploration, allowing users to identify warming trends, variability patterns, and the frequency of extreme temperature events.

**Sea Surface Temperature (SST)** refers to the temperature of the upper layer of the ocean, typically measured within the top 1 mm to 10 meters.
Tracking SST anomalies over time helps identify long-term warming trends and unusual ocean temperature events.

## Dataset
The project uses the NOAA Optimum Interpolation (OI) SST V2 [dataset](https://www.psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html). The dataset covers Sea Surface Temperature from 1990 to 2022 with weekly and biweekly resolution.


## Running the Dashboard
You will need to run the Dasboard inside a conda environment.
```
cd MHW-Detection
conda env create -f environment.yml
conda activate mhw-detection
```
Once the installation done you can run the dasboard with the following:
```
cd MHW-Detection
panel serve app/interactive_map_panel.py --show
```

## Features

The dashboard provides a spatial-to-temporal workflow:
- **Variability Map** (Left Panel):
    The map displays regions of high or low SST variability across years.
    This map is computed by:
    - Calculating SST anomalies
    - Computing the standard deviation of anomalies for each month
    - Averaging these values across months

      >**Warning:**
      >This map highlights where SST fluctuates most from year to year and should not be interpreted as an ‘extreme’ or ‘heatwave’ map.


- **Interactive Location Analysis** (Right Panel):
    Clicking on a location generates several analyses:
    - Linear Trend Estimation: An Ordinary Least Squares (OLS) regression is applied to estimate the long-term SST anomaly trend. This provides a simple statistical estimate of local ocean warming or cooling.
    
    - Rolling Mean Visualization: A 3-year rolling mean is computed to visualize long-term SST evolution. This helps smooth short-term variability and makes long-term changes easier to observe.

        > **Note**:
        > The rolling mean is a visualization aid, not a statistical estimator.

    - Extreme Event Detection: 
    The project initially aimed to detect Marine Heatwaves (MHW).
    However, standard marine heatwave detection requires daily SST data, while the dataset used here is weekly or biweekly.
    Instead, the dashboard detects Extreme Events, defined as SST anomalies exceeding a 95th percentile threshold:
    This allows the analysis of whether warm anomalies are becoming more frequent and how the distribution of SST anomalies evolves over time


## References

- Reynolds, R.W., N.A. Rayner, T.M. Smith, D.C. Stokes, and W. Wang, 2002: An improved in situ and satellite SST analysis for climate. J. Climate, 15, 1609-1625.
