from pathlib import Path
import sys
from scipy.stats import gaussian_kde

import hvplot.pandas
import hvplot.xarray
import os 
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from data import *


pn.extension()
hv.extension("bokeh")
xr.set_options(display_style="text")


def _resolve_data_path(path_from_config):
    # config paths are relative to src/, so resolve from that base.
    return (SRC_DIR / path_from_config).resolve()

@pn.cache
def load_masked_dataset(filename,engine="netcdf4"):
    data_path = _resolve_data_path(filename)

    ds = xr.open_dataset(
        data_path,
        engine=engine,
        chunks=config.CHUNKS,
    )
    return ds

@pn.cache
def load_masked_zarr_dataset(filename):
    data_path = _resolve_data_path(filename)

    ds = xr.open_zarr(
        data_path,
    )
    return ds


@pn.cache
def load_initial_map():
    cache_path = _resolve_data_path(config.INITIAL_MAP_CACHE)
    if cache_path.exists():
        try:
            return xr.open_dataarray(cache_path)
        except Exception:
            pass

    ds = load_masked_dataset(config.DATA_PATH)
    ds["time"] = xr.decode_cf(ds).time
    sst = ds.sst
    sst = sst.sel(time=slice(config.MIN_DATE, config.MAX_DATE))
    if config.MAP_COARSEN and config.MAP_COARSEN > 1:
        sst = sst[::config.TIME_COARSEN, ::config.MAP_COARSEN, ::config.MAP_COARSEN]
    sst_grouped = sst.groupby("time.month")
    tos_std = sst_grouped.std(dim="time")
    initial_map = tos_std.mean(dim="month")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        initial_map.astype("float32").to_netcdf(cache_path)
    except Exception:
        pass

    return initial_map


def compute_anomaly_plot(df: pd.DataFrame, tos_anom_selected: xr.DataArray, lon:float, lat:float):
    rolling_year_num = config.ROLLING_YEARS
    sme_step = int(config.FREQ_PER_YEAR_MIN)

    # anom_resampled = tos_anom_selected# tos_anom_selected.resample(time="SME").mean("time")
    rolling_year_avg = tos_anom_selected.rolling(
        time=sme_step * rolling_year_num, center=True
    ).mean()

    df_clean = df.dropna().copy()
    X = df_clean["continuous_time"].to_numpy().reshape(-1, 1)
    y = df_clean["anomalies"].to_numpy().reshape(-1, 1)

    reg = LinearRegression().fit(X, y)
    trend = np.round(reg.coef_[0, 0] * 10, 3)
    df_clean["pred_ols"] = reg.predict(X)[:, 0]

    resampled_df = pd.DataFrame(
        {
            "rolled_time": tos_anom_selected.time.values,
            "anomalies_rolled_avg": rolling_year_avg.values,
        }
    )

    anomaly_time_plot = tos_anom_selected.hvplot.line(
        x="time",
        y="sst",
        title=(
            f"Anomaly at selected location "
            f"({lon:.2f}, {lat:.2f}) "
            f"| Trend: {trend:.3f} ˚C/decade"
        ),
        width=config.RIGHT_PANEL_WIDTH,
        height=config.RIGHT_PLOT_HEIGHT,
        color="#2b8cbe",
        line_width=1.8,
        label="Anomaly",
        ylabel="SST anomaly (˚C)",
        xlabel="Time (months)",
    ).opts(active_tools=["pan"])

    trend_plot = df_clean.hvplot.line(
        x="time",
        y="pred_ols",
        color="#ff8c00",
        width=config.RIGHT_PANEL_WIDTH,
        height=config.RIGHT_PLOT_HEIGHT,
        line_width=2.3,
        label="OLS trend",
    )

    rolling_avg_plot = resampled_df.hvplot.line(
        x="rolled_time",
        y="anomalies_rolled_avg",
        color="#d7301f",
        width=config.RIGHT_PANEL_WIDTH,
        height=config.RIGHT_PLOT_HEIGHT,
        line_width=2.6,
        label=f"{rolling_year_num}-year rolling mean",
    ).opts(active_tools=["pan"])

    return (anomaly_time_plot * trend_plot * rolling_avg_plot).opts(
        shared_axes=False, legend_position="top_left", legend_cols=3, show_grid=True,
            legend_opts={"border_line_alpha": 0.0, "label_text_font_size": '10px',"margin": 0,}
    )


def compute_extreme_events_plot(df: pd.DataFrame):
    quantile_val = config.EXTREME_QUANTILE

    df_extreme = df.loc[df["q95"] == 1]
    thr_value = np.quantile(df["anomalies"], q=quantile_val)

    anomaly_curve = df.hvplot.line(
        x="time",
        y="anomalies",
        color="gray",
        alpha=0.6,
        width=config.RIGHT_PANEL_WIDTH,
        height=config.RIGHT_PLOT_HEIGHT,
        ylabel="SST anomaly (˚C)",
        xlabel="Time (months)",
        title=f"Extreme events above {int(quantile_val * 100)}th percentile threshold",
        label="Anomaly",
    )

    thr_line = hv.HLine(thr_value).opts(
        color="red",
        line_dash="dashed",
        line_width=2,
    ).relabel(f"q{int(quantile_val * 100)}")

    extreme_event = df_extreme.hvplot.scatter(
        x="time",
        y="anomalies",
        color="red",
        size=45,
        alpha=0.9,
        label="Extreme events",
    )

    return (anomaly_curve * thr_line * extreme_event).opts(
        active_tools=["pan"], legend_position="top_left", legend_cols=3, show_grid=True,
        legend_opts={"border_line_alpha": 0.0, "label_text_font_size": '10px',"margin": 0,}

    )


def compute_barplot(df: pd.DataFrame):
    df_sel = df.loc[df["q95"] == 1]
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())

    full_years = pd.Index(np.arange(year_min, year_max + 1), name="year")
    counts = df_sel.groupby("year")["q95"].sum().reindex(full_years, fill_value=0)
    df_bar = counts.rename("number of extreme events").reset_index()

    tick_step = 5
    tick_values = np.arange(year_min, year_max + 1, tick_step)

    bar_plot = df_bar.hvplot.bar(
        x="year",
        y="number of extreme events",
        alpha=0.5,
        line_alpha=0.5,
        bar_width=0.8,
        title="Number of Extreme Events",
        width=config.RIGHT_PANEL_WIDTH,
        height=config.RIGHT_PLOT_HEIGHT,
        color="#74a9cf",
        xlabel="Time (years)",
        ylabel="Number of extreme events",
    ).opts(active_tools=["pan"], show_grid=True
)

    # Expand years by count for KDE fit
    expanded_years = np.repeat(df_bar["year"].values, df_bar["number of extreme events"].values)

    if expanded_years.size >= 2:
        kde = gaussian_kde(expanded_years, bw_method=0.35)  # tune bandwidth
        x_grid = np.linspace(year_min, year_max, 400)
        y_kde = kde(x_grid)

        # Scale KDE so it is visually comparable to counts
        y_kde_scaled = y_kde * (df_bar["number of extreme events"].max() / y_kde.max())

        kde_curve = hv.Curve(
            (x_grid, y_kde_scaled), 
            "year", 
            "kde_scaled",
            label='kde').opts(
            color="#74a9cf", line_width=2.5
        )
        bar_plot *= kde_curve

    return bar_plot.opts(show_grid=True,legend_position="top_left", 
                         legend_opts={"border_line_alpha": 0.0, "label_text_font_size": '10px',"margin": 0,})

def build_raw_timeseries_view():
    ds_ssta = load_masked_dataset(config.ANOMALY_MAP_PATH, engine='zarr')
    ssta = ds_ssta.sst
    # if config.RAW_COARSEN and config.RAW_COARSEN > 1:
    #     ssta = ssta[::config.TIME_COARSEN, ::config.RAW_COARSEN, ::config.RAW_COARSEN]

    raw_map = ssta.hvplot(
        x="lon",
        y="lat",
        cmap="RdBu_r",
        groupby="time",
        width=config.TIME_SERIE_WIDTH,
        height=config.TIME_SERIE_HEIGHT,
        clim=(-5, 5),
        clabel="SST anomaly (˚C)",
        title="Weekly SST anomaly",
        xlabel="Longitude (degrees_east)",
        ylabel="Latitude (degrees_north)",
        widget_location="bottom",
    )

    text = """
        The **worldmap** shows weekly sea surface temperature (SST) **anomalies** — deviations from the monthly climatology — across the global ocean. The diverging colormap is centered on zero: **red** indicates warmer-than-usual water, **blue** cooler-than-usual, and **white** near-normal conditions. The color scale is clipped to ±5 ˚C.

        - **Time slider**: drag to scan through weekly frames; synced to the dataset time coordinate.
        - **Pan & zoom**: use the Bokeh toolbar to inspect regional structure.
        - **Grid**: 0.25˚ × 0.25˚ (coarsened for interactive performance).
    """
    note = pn.pane.Markdown(text, sizing_mode="stretch_width")

    return pn.Column(raw_map, note, sizing_mode="stretch_both")


def build_anomaly_view():
    ds_ssta = load_masked_dataset(config.ANOMALY_MAP_PATH, engine="zarr")
    initial_map = load_initial_map()

    initial_plot = initial_map.hvplot(
        x="lon",
        y="lat",
        cmap="OrRd",
        title="Sea Surface Temperature Variability across years",
        width=config.MAP_WIDTH,
        height=config.MAP_HEIGHT,
        xlabel="Longitude (degrees_east)",
        ylabel="Latitude (degrees_north)",
    ).opts(active_tools=["pan"], show_grid=False)

    posxy = hv.streams.Tap(
        source=initial_plot, x=config.DEFAULT_TAP_LON, y=config.DEFAULT_TAP_LAT
    )

    def select_point(x, y):
        tos_anom_selected = ds_ssta.sst.sel(lon=x, lat=y, method="nearest")

        df = pd.DataFrame(
            {
                "anomalies": tos_anom_selected.values,
                "year": tos_anom_selected.time.dt.year.values,
                "time": tos_anom_selected.time.values,
            }
        )
        df["continuous_time"] = (
            tos_anom_selected.time.dt.year
            + (tos_anom_selected.time.dt.dayofyear - 1) / 365.25
        ).values

        q95 = np.quantile(df["anomalies"].dropna(), q=config.EXTREME_QUANTILE)
        df["q95"] = (df["anomalies"] > q95).astype(int)

        anomaly_time_plot = compute_anomaly_plot(df, tos_anom_selected, x, y)
        extreme_event_plot = compute_extreme_events_plot(df)
        bar_plot = compute_barplot(df)

        return pn.Column(
            anomaly_time_plot,
            extreme_event_plot,
            bar_plot,
            sizing_mode="stretch_width",
            width=config.RIGHT_PANEL_WIDTH,
        )

    text = """
    The above **worldmap** highlights where Sea Surface Temperature (SST) fluctuates most from year to year (red). Clicking on a location generates several analyses: 

    - **Linear trend estimation**: An Ordinary Least Squares (OLS) regression is applied to estimate the long-term SST anomaly trend.
    - **Extreme event detection**: defined as SST anomalies exceeding a 95th percentile threshold
    - **Histogram of the number of extreme event** with Kernel Density Estimation (KDE): detects if warm anomalies are becoming more frequent and how the distribution of SST anomalies evolves over time.
    """ 
    caption_text = pn.pane.Markdown(text)
    right_panel = pn.bind(select_point, x=posxy.param.x, y=posxy.param.y)
    dashboard = pn.Row(
        pn.Column(initial_plot, caption_text, sizing_mode="stretch_width"),
        pn.Spacer(width=12),
        right_panel,
        sizing_mode="stretch_width",
    )

    return dashboard


def build_mhw_view():
    ds = load_masked_dataset(config.MHW_MAP_PATH, engine='zarr')
    max_days = float(np.nanmax(ds.day_per_year.values))
    max_event = float(np.nanmax(ds.event_per_year.values))

    selector = pn.widgets.Select(
        name="Metric",
        options={"Days per year": "day_per_year", "Events per year": "event_per_year"},
        value="day_per_year",
        width=220,
    )
    allowed_values = [int(y) for y in ds.year.values]  # avoid numpy int64 in widget

    year_slider = pn.widgets.DiscreteSlider(
        name="Year",
        options=allowed_values,
        value=allowed_values[0],
    )

    note = pn.pane.Markdown(
        """
        **Marine Heatwave (MHW)** events are defined as periods of ≥5 consecutive days where
        sea surface temperature exceeds the 90th-percentile climatological threshold for that
        calendar day (±5-day window). The map shows either the total number of MHW **days** or
        discrete MHW **events** per year at each grid cell.

        - Click anywhere on the map to see the full year-by-year time series at that location.
        - Use the metric selector and year slider to explore spatial patterns.
        """,
        sizing_mode="stretch_width",
    )

    # Reactive store for the Tap stream — updated whenever the map redraws
    tap_stream = hv.streams.Tap(x=config.DEFAULT_TAP_LON, y=config.DEFAULT_TAP_LAT)

    def _map(metric, year):
        max_val = max_event if metric == "event_per_year" else max_days
        da = ds[metric].sel(year=year)
        plot = da.hvplot(
            x="lon",
            y="lat",
            width=config.MAP_WIDTH,
            height=config.MAP_HEIGHT,
            clim=(0, max_val),
            cmap="inferno",
            title=f"MHW {metric.replace('_', ' ')} ({year})",
        ).opts(active_tools=["pan"])
        tap_stream.source = plot
        return plot

    def _timeseries(metric, x, y):
        ylabel = "Days per year" if metric == "day_per_year" else "Events per year"
        da_point = ds[metric].sel(lon=x, lat=y, method="nearest")
        actual_lon = float(da_point.lon)
        actual_lat = float(da_point.lat)
        df_ts = pd.DataFrame({
            "year": [int(yr) for yr in da_point.year.values],
            ylabel: da_point.values,
        }).dropna()

        year_min, year_max = df_ts["year"].min(), df_ts["year"].max()

        bar_plot = df_ts.hvplot.bar(
            x="year",
            y=ylabel,
            alpha=0.5,
            line_alpha=0.5,
            bar_width=0.8,
            title=f"MHW {ylabel} at ({actual_lon:.2f}°, {actual_lat:.2f}°)",
            width=config.MAP_WIDTH,
            height=config.RIGHT_PLOT_HEIGHT,
            color="#74a9cf",
            xlabel="Year",
            ylabel=ylabel,
        ).opts(active_tools=["pan"], show_grid=True)

        expanded = np.repeat(df_ts["year"].values, df_ts[ylabel].values.astype(int).clip(0))
        if expanded.size >= 2:
            kde = gaussian_kde(expanded, bw_method=0.35)
            x_grid = np.linspace(year_min, year_max, 400)
            y_kde = kde(x_grid)
            y_kde_scaled = y_kde * (df_ts[ylabel].max() / y_kde.max())
            kde_curve = hv.Curve(
                (x_grid, y_kde_scaled), "year", "kde_scaled", label="kde",
            ).opts(color="#74a9cf", line_width=2.5)
            bar_plot *= kde_curve

        return bar_plot.opts(
            show_grid=True, legend_position="top_left",
            legend_opts={"border_line_alpha": 0.0, "label_text_font_size": "10px", "margin": 0},
        )

    map_panel = pn.bind(_map, metric=selector, year=year_slider)
    ts_panel = pn.bind(_timeseries, metric=selector, x=tap_stream.param.x, y=tap_stream.param.y)

    controls = pn.Column(selector, year_slider, note, sizing_mode="stretch_width", width=260)
    return pn.Row(
        controls,
        pn.Column(map_panel, ts_panel, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

def build_app():
    tabs = pn.Tabs(
        ("SST Anomalies (Time Slider)", build_raw_timeseries_view()),
        ("Anomaly Explorer", build_anomaly_view()),
        ("Marine HeatWave Visualization", build_mhw_view()),
        tabs_location="left",
        sizing_mode="stretch_both",
        dynamic=True,
    )

    return pn.template.FastListTemplate(
        title="SST Dashboard",
        main=[tabs],
        accent_base_color="#0d6e6e",
        header_background="#0d6e6e",
    )


app = build_app()
app.servable()


if __name__ == "__main__":
    pn.serve(app, show=True, title="SST Anomaly Explorer")
