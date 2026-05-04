from pathlib import Path
import sys
from scipy.stats import gaussian_kde

import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from sklearn.linear_model import LinearRegression
import bokeh.palettes

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config


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
def load_initial_map():
    cache_path = _resolve_data_path(config.INITIAL_MAP_CACHE)
    if cache_path.exists():
        try:
            return xr.open_dataarray(cache_path)
        except Exception:
            pass

    ds = load_masked_dataset(config.DATA_PATH)
    ds["time"] = xr.decode_cf(ds).time
    sst = ds.sst.sel(time=slice(config.MIN_DATE, config.MAX_DATE))
    if config.MAP_COARSEN and config.MAP_COARSEN > 1:
        sst = sst[::config.TIME_COARSEN, ::config.MAP_COARSEN, ::config.MAP_COARSEN]
    sst_grouped = sst.groupby("time.month")
    tos_std = sst_grouped.std(dim="time")
    initial_map = tos_std.mean(dim="month")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        initial_map.astype("float32").to_zarr(cache_path, mode='w', consolidated=True)
    except Exception:
        pass

    return initial_map


def compute_anomaly_plot(df: pd.DataFrame, tos_anom_selected: xr.DataArray, lon:float, lat:float):
    rolling_year_num = config.ROLLING_YEARS
    sme_step = int(config.FREQ_PER_YEAR_MIN)

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
    ).opts(active_tools=["pan"], show_grid=True)

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
    video_path = _resolve_data_path(config.SSTA_VIDEO_PATH)

    video = pn.pane.Video(
        str(video_path),
        loop=True,
        sizing_mode="stretch_width",
        height=config.TIME_SERIE_HEIGHT,
    )

    text = """
        The **video** shows weekly sea surface temperature **anomalies** (SSTA). 
        SSTA are deviations from the monthly climatology across the global ocean. The diverging colormap is centered on zero: 
        **red** indicates warmer-than-usual water, **blue** cooler-than-usual, and **white** near-normal conditions. 
        The color scale is clipped to ±5 ˚C. We can see a clear increase in appearance of hot events with higher probabilities, spread, strength.

        - **Play/Pause**: use the video controls to start or stop playback.
        - Frames between real weekly timesteps are linearly interpolated for smooth motion.
        - Speed can also be adjusted.
    """
    note = pn.pane.Markdown(text, sizing_mode="stretch_width")

    return pn.Column(video, note, sizing_mode="stretch_both")


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
        The above **worldmap** highlights where Sea Surface Temperature (SST) fluctuates most 
        from year to year (red). Clicking on a location generates several analyses: 

        - **Linear trend estimation**: An Ordinary Least Squares (OLS) regression is applied 
            to estimate the long-term SST anomaly trend.
        - **Extreme event detection**: defined as SST anomalies exceeding a 95th percentile threshold
        - **Histogram of the number of extreme event** with Kernel Density Estimation (KDE): 
            detects if warm anomalies are becoming more frequent and how the distribution of
            SST anomalies evolves over time.

        **Analysis**:
        Areas showing strong variability independently of seasonality are:
        
        - **El Niño-Southern Oscillation (ENSO):** Recurring climate pattern (3 to 7 years) 
            warming (El Niño) or cooling (La Niña) the water by ~1 to 3 C compared to normal SST.
            ENSO is one of the most important climate phenomena on Eart as it changes temperatures, 
            precipitation around the globe.
        
        - **Gulf Stream (GS):** A strong ocean current bringing warm water from the Gulf of Mexico 
            into the Atlantic ocean. Hence this location has more variability than normal. 
            The cold water from the Labrador current goes down and the warmer than normal SSTs stick around. 
            Anomalies are computed relative to the mean, a small displacement of the current can 
            produces a larger deviation from the mean.
        
        - **Kuroshio Extension (KE):** Similar to the GS, the KE is a powerful current in the Pacific 
            from the East Coast of the Japan and meanders east towards North America. 
        
        - **Agulhas current (AC):** Strong ocean current bringing also warm water from the southeast 
            coast of Mozambique to South Africa and then meanders east toward Australia.
        
        - **Brazil-Malvinas (or Falkland) Confluence (BMC):** Confluence of 2 currents off the coast of 
            Argentina and Uruguay where the warm Brazil Current and the cold Falkland current converge.

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
        width=config.MHW_SLIDER_WIDTH,
    )
    allowed_values = [int(y) for y in ds.year.values]

    year_slider = pn.widgets.DiscreteSlider(
        name="Year",
        options=allowed_values,
        value=allowed_values[0],
        width=config.MHW_SLIDER_WIDTH,
    )

    note = pn.pane.Markdown(
        """
        This panel detects **Marine HeatWave (MHW)** events defined as periods of **≥5 consecutive days** 
        where SST exceeds the 90th-percentile climatological threshold. The map shows either the total 
        number of MHW **days** or discrete MHW **events** per year at each grid cell.

        - Click anywhere on the map to see the full time series per year at that location.
        - Use the metric selector and year slider to explore spatial patterns.


        **Analysis**: 

        Several recurring patterns emerge across the maps, intensifying from the 1980s to the 2020s:
        
        - **1983, 1997-1998, 2015-2016, and 2023-2024** correspond to major El Niño events and appear as the brightest, 
            most spatially extensive maps. El Niño also indirectly warms the Indian and Atlantic Oceans later by 
            reorganizing global wind patterns. 2016 combines a strong El Niño with the long-term warming trend, 
            making it the most intense map in the series. 
        
        - **Long-term background warming:** Comparing **1987** and  **2022** two quiet years without any ENSO 
            phenomenon reveals a critical shift: the 2022 map is considerably brighter almost everywhere. 
            This illustrates the human impact on ocean warming, gradually raising the baseline temperature of the ocean. 
            As a result, even ordinary years now produce more MHW days (2022) than exceptional years did 40 ago (1983).
        
        - **1992**: A localized MHW signal appears in the Southern Ocean between Australia and South America. 
            The 1992 anomaly is likely linked to the **1991 Mount Pinatubo volcanic eruption**, which disrupted 
            global wind patterns and may have caused regional warming in this area.
        
        - **2005, 2011-2012**: The Arctic is the fastest-warming region on the planet because melting white ice 
            (reflective) is replaced with dark, heat-absorbing ocean water, which accelerates further melting. 
            This makes the Arctic particularly sensitive to MHW conditions. Whether these specific years were 
            triggered by discrete events or by the long-term warming trend remains unclear.
        \n

        The 1982-2025 period shows that **MHWs that were once rare, confined, and tied to El Niño** events are **now 
        longer, more widespread and frequent** even in the absence of any specific climate phenomenon. 
        
        **This highlights an ongoing and accelerating warming of the global ocean.**

        """,
        sizing_mode="stretch_width",
        width=config.MHW_SLIDER_WIDTH
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
            cmap="afmhot",
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

    controls = pn.Column( pn.Row(selector, year_slider), note, width=config.RIGHT_PANEL_WIDTH)
    return pn.Row(
        controls,
        pn.Column(map_panel, ts_panel, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

def build_forecast_view():
    metric_path = _resolve_data_path(config.FORECAST_ACC_PATH)
    forecast_path = _resolve_data_path(config.FORECAST_CHART_PATH)

    if not metric_path.exists() or not forecast_path.exists():
        return pn.pane.Markdown(
            """
            **Forecast dataset not found** \n
            Please see [the README.md](https://github.com/lucieDLE/MHW-Detection?tab=readme-ov-file#training-a-model) for more information on how to generate this file.
            """,
            sizing_mode="stretch_width", renderer='markdown'
        )

    metric_ds = xr.open_zarr(str(metric_path))
    forecast_ds = xr.open_zarr(str(forecast_path))

    model_name   = metric_ds.model
    input_window = metric_ds.input_window
    rmse_max     = float(metric_ds["rmse_range"].values[1]) if "rmse_range" in metric_ds else 2.0
    lead_times   = [int(v) for v in np.array(config.LEAD_TIMES)]
    anchor_dates = [str(t)[:10] for t in forecast_ds.anchor_time.values]

    text_forecast = """
        **Forecasting skill**
        
        - **Tropical band**: Across almost all lead times, the model captures SST dynamics most reliably in the ENSO region.
            Skill remains positive up to t = 28 days, reflecting the slow, large-scale nature of ENSO-driven anomalies.
        
        - **High latitudes**: The model underperforms persistence in the Arctic and Southern Ocean, increasingly with time.
            High-latitude SST is driven by fast, chaotic processes (storms, sea ice dynamics, turbulent mixing), where small
            initial errors are carried forward and amplified at each rollout step.
        
        - **Weddell Sea**: The persistently deep blue spot in the Southern Ocean is the worst-performing
            region across all lead times, including at t = 1. This area is called the **Weddell Sea**, which is governed
            by sea ice melt and freeze cycles that are fundamentally different from open ocean dynamics.
    """

    text_RMSE = """
        **RMSE Maps (Model and Persistence)**
        
        - Both the model and persistence share a very similar spatial error structure: highest RMSE along strong ocean
            currents (Gulf Stream, Kuroshio Extension), in high-latitude regions, and near coastlines.
        
        - This confirms that 1) the model has learned a level of predictability comparable to persistence at short lead times and
            2) the regions that are hardest to forecast are the same for both methods. These areas are known to have high SST variability.
        
        - By lead time 21, the model's RMSE deteriorates significantly in the most dynamically active regions,
            while the tropical Pacific and Indian Ocean remain comparatively well-forecasted.
            This is a direct consequence of the rollout error accumulation in high-variability areas.
    """

    text_ACC = """
        **ACC Maps (Model and Persistence)**
        
        - Both methods maintain high ACC (deep red, close to 1) across most of the global ocean at all lead times. However, 
        this is expected: ACC measures correlation, and SST anomalies change slowly, making them easier to predict over short 
        to medium horizons, and therefore easier to correlate with observations.
        
        - The most informative regions are where the maps fade from red toward yellow/white: strong currents and high-latitude areas 
        associated with high and fast SST variability. By t = 28 days, the model appears to achieve slightly better performance 
        in these regions, but the difference is very subtle. 
        
        - This is why the ACC difference maps are more informative than these absolute ACC maps alone.
    
    """

    text_diff_ACC = """
        **ACC Difference Map**

        - Up to t = 10 days: The ACC difference is mostly neutral globally, with a clear positive band along the
            equatorial Pacific confirming that the model outperforms persistence in the ENSO region.
            High latitudes show a mild negative signal but remain close to zero.
        
        - t = 14 to t = 28 days: The spatial structure degrades rapidly. The Arctic and Southern Ocean turn strongly blue,
            with the model losing up to 0.4 ACC points relative to persistence.
        
        - At t = 28 days: The model adds very little value over persistence at the global scale, with meaningful positive skill
            confined almost entirely to the tropical Pacific.
    """


    metric_description = """
        **Metrics**
        
        - **Anomaly Correlation Coefficient (ACC)**: Spatial Pearson correlation between predicted and observed SSTA per timestep,
            averaged over the test period.
        
        - **Root Mean Square Error (RMSE)**: Per-pixel error between predicted and observed SSTA.
        
        - **Forecasting Skill Score**: Measures improvement over the persistence baseline.
            A score of 0 means the model performs no better than persistence; 1 means a perfect forecast; a negative score means persistence wins.
            blue regions indicate where persistence is more informative.

        **Overall**: The model has genuinely learned useful structure in the tropics, where slow ENSO-driven signals provide a learnable target.
        However, the autoregressive rollout strategy is too fragile for high-latitude regions, where errors compound fastest.
        A direct model (n_out = 28) would likely address much of this degradation without requiring a more complex architecture.

        You can use the select option to dive into model performances and comparisons:
        
        - Select a metric and lead time to display the corresponding spatial map, accompanied by an analysis.
        
        - Select a time period and click anywhere on the map to visualize the predicted SSTA at that location.
    """
    metric_description_note = pn.pane.Markdown(metric_description)

    metric_options = {
        f"{model_name} ACC": "model_acc",
        "Persistence ACC": "persistence_acc",
        "ACC Difference": "acc_diff",
        f"{model_name} RMSE": "model_rmse",
        "Persistence RMSE": "persistence_rmse",
        "Forecasting skill": "forecast_skill",
    }

    metric_selector = pn.widgets.Select(name="Metric", options=metric_options, value="model_acc", width=280,)
    lead_slider = pn.widgets.DiscreteSlider(name="Lead time (days)", options=lead_times, value=lead_times[0], width=280)
    anchor_slider = pn.widgets.DiscreteSlider(name="Forecast start date", options=anchor_dates, value=anchor_dates[len(anchor_dates) // 2], width=280)

    note = pn.pane.Markdown(
        f"**Input window:** {input_window} days  | **Test period:** " + str(config.DL_TEST_RANGE) + "\n\n",
        sizing_mode="stretch_width",
    )

    tap_stream = hv.streams.Tap(x=config.DEFAULT_TAP_LON, y=config.DEFAULT_TAP_LAT)

    def _plot(metric_key, lead):
        if metric_key == "acc_diff": # build the ACC difference
            da = metric_ds[f"model_acc"].sel(lead_time=lead) - metric_ds[f"persistence_acc"].sel(lead_time=lead)
        elif metric_key == "forecast_skill": # build the skill metric
            da = 1 - (metric_ds[f"model_rmse"].sel(lead_time=lead) / metric_ds[f"persistence_rmse"].sel(lead_time=lead))
        else:
            da = metric_ds[metric_key].sel(lead_time=lead)

        is_acc  = "acc"  in metric_key
        is_diff = "diff" in metric_key

        if "acc" in metric_key:
            cmap, clim = ("RdYlBu_r", (-1, 1)) if not is_diff else ("RdBu_r", (-0.5, 0.5))
        elif 'rmse' in metric_key:
            cmap, clim = ("YlOrRd", (0, rmse_max)) if not is_diff else ("RdBu_r", (-rmse_max / 2, rmse_max / 2))
        else:
            ncolors=256
            clim = (-5, 1)
            normalized_mid_point = (0 - clim[0]) / (clim[1] - clim[0])
            cmap = bokeh.palettes.diverging_palette(bokeh.palettes.Blues[ncolors], 
                                                      bokeh.palettes.Reds[ncolors], 
                                                      n=ncolors, 
                                                      midpoint=normalized_mid_point)

        label = [k for k, v in metric_options.items() if v == metric_key][0]
        plot = da.hvplot(
            x="lon", y="lat",
            cmap=cmap,clim=clim,
            width=config.MAP_WIDTH,
            height=config.MAP_HEIGHT,
            title=f"{label}  —  lead time = {lead} days",
            xlabel="Longitude", ylabel="Latitude",
        ).opts(active_tools=["pan"])
        tap_stream.source = plot
        return plot

    def _forecast_chart(x, y, anchor_date):
        if forecast_ds is None:
            return pn.pane.Markdown("*Fan chart not available — re-run `export_to_dataset.py` to generate.*")

        # select the curve from anchor data to -14 to +28 days for the selected locations
        anchor_t = np.datetime64(anchor_date)
        context_window  = forecast_ds.input_context.sel(anchor_time=anchor_t, lon=x, lat=y, method="nearest").values  # (n_in,)
        pred = forecast_ds.model_pred.sel(anchor_time=anchor_t, lon=x, lat=y, method="nearest").values     # (horizon,)
        truth = forecast_ds.truth.sel(anchor_time=anchor_t, lon=x, lat=y, method="nearest").values         # (horizon,)

        n_in_v    = int(forecast_ds.input_window)
        horizon_v = int(forecast_ds.horizon)

        input_dates    = [anchor_t + np.timedelta64(d, 'D') for d in range(-n_in_v + 1, 1)] # t-14 to t
        forecast_dates = [anchor_t + np.timedelta64(d, 'D') for d in range(1, horizon_v + 1)] # t to t+28

        persistence_val = float(context_window[-1])
        persistence = np.full(horizon_v, persistence_val)

        df_obs = pd.DataFrame({"date": input_dates,    "ssta": context_window,         "series": "observed (input)"})
        df_truth = pd.DataFrame({"date": forecast_dates, "ssta": truth,     "series": "observed (truth)"})
        df_model = pd.DataFrame({"date": forecast_dates, "ssta": pred,      "series": f"{model_name} forecast"})
        df_pers  = pd.DataFrame({"date": forecast_dates, "ssta": persistence,"series": "persistence"})

        df = pd.concat([df_obs, df_truth, df_model, df_pers], ignore_index=True)

        color_map = {
            "observed (input)":       "#2b8cbe",
            "observed (truth)":       "#2b8cbe",
            f"{model_name} forecast": "#d7301f",
            "persistence":            "#ff8c00",
        }
        dash_map = {
            "observed (input)":       "solid",
            "observed (truth)":       "dashed",
            f"{model_name} forecast": "dashed",
            "persistence":            "dotted",
        }

        actual_lon = float(forecast_ds.model_pred.sel(anchor_time=anchor_t, lon=x, lat=y, method="nearest").lon)
        actual_lat = float(forecast_ds.model_pred.sel(anchor_time=anchor_t, lon=x, lat=y, method="nearest").lat)

        plots = []
        for label, grp in df.groupby("series", sort=False):
            plots.append(
                grp.hvplot.line(
                    x="date", y="ssta",
                    label=label,
                    color=color_map[label],
                    line_dash=dash_map[label],
                    line_width=2.2,
                    width=config.MAP_WIDTH,
                    height=config.RIGHT_PLOT_HEIGHT,
                )
            )

        anchor_line = hv.VLine(anchor_t).opts(color="gray", line_dash="dashed", line_width=1.2)

        return (hv.Overlay(plots) * anchor_line).opts(
            title=f"Forecast trajectories at ({actual_lon:.1f}°, {actual_lat:.1f}°)  —  start: {anchor_date}",
            xlabel="Date", ylabel="SSTA (°C)",
            legend_position="top_left",
            show_grid=True,
            active_tools=["pan"],
        )


    _metric_to_analysis = {
        "model_acc":       text_ACC,
        "persistence_acc": text_ACC,
        "model_rmse":      text_RMSE,
        "persistence_rmse":text_RMSE,
        "acc_diff":        text_diff_ACC,
        "forecast_skill":  text_forecast,
    }

    def _analysis_text(metric_key):
        return pn.pane.Markdown(_metric_to_analysis[metric_key], sizing_mode="stretch_width")

    analysis_panel = pn.bind(_analysis_text, metric_key=metric_selector)

    map_panel = pn.bind(_plot, metric_key=metric_selector, lead=lead_slider)
    forecast_panel = pn.bind(_forecast_chart, x=tap_stream.param.x, y=tap_stream.param.y, anchor_date=anchor_slider)
    
    centered_slider = pn.Row(pn.HSpacer(), anchor_slider, pn.HSpacer())
    controls = pn.Column(note, 
                         pn.Row(metric_selector, lead_slider),
                         metric_description_note, 
                         analysis_panel, 
                         width=config.RIGHT_PANEL_WIDTH)

    return pn.Row(controls, pn.Column(map_panel, forecast_panel,centered_slider, sizing_mode="stretch_width"), sizing_mode="stretch_width")


def build_app():
    tabs = pn.Tabs(
        ("SST Anomalies (Video)", build_raw_timeseries_view()),
        ("Anomaly Explorer", build_anomaly_view()),
        ("Marine HeatWave Visualization", build_mhw_view()),
        ("SST Forecasting", build_forecast_view()),
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
