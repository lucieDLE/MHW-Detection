import path_setup  # noqa: F401  sets up sys.path; import first

import functools

import config
import theme
import xarray as xr

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression


# ============================================================================
#  FIGURES AND DATA FUNCTIONS
# ============================================================================

@functools.lru_cache(maxsize=None)
def open_zarr(path_str):
    return xr.open_zarr(path_str)


def heatmap_fig(z, lons, lats, colorscale, title, zmin=None, zmax=None):
    fig = go.Figure(go.Heatmap(
        z=z, x=lons, y=lats,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(thickness=12, len=0.9),
    ))
        
    # Template + backgrounds are owned by theme.apply_theme (called per callback).
    fig.update_layout(title=title, **theme.MAP_LAYOUT)
    return fig


def add_kde_overlay(fig, expanded, x_min, x_max, peak, bw_method=0.35, n_points=400):
    """Overlay a smoothed-trend curve on a yearly-count bar chart.

    ``expanded`` is the count-weighted sample (years repeated by their count).
    A Gaussian KDE of that sample is rescaled so its peak matches ``peak`` (the
    tallest bar), so it sits on the bar axis. This is a *visual* smoothing to
    show how events cluster over time — not a calibrated frequency model: the
    sample is discrete (integer years) and the curve is rescaled, so its height
    is not a density. No-op if there are fewer than 2 samples.
    """
    if expanded.size < 2:
        return fig
    kde = gaussian_kde(expanded, bw_method=bw_method)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_kde = kde(x_grid)
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_kde * (peak / y_kde.max()),
        mode="lines", name="Smoothed trend", line=dict(color=theme.RED_LIGHT, width=2.5),
    ))
    return fig


# ============================================================================
#  ANOMALIES FIGURES
# ============================================================================

def initial_map_fig():
    ds= open_zarr(config.INITIAL_MAP_CACHE)
    da = ds.sst
    return heatmap_fig(
        z=da.values, lons=da.lon.values, lats=da.lat.values,
        colorscale="OrRd",
        title="Sea Surface Temperature Variability across years",
    )

def anomaly_figs(lon, lat):
    ds_ssta = open_zarr(config.ANOMALY_MAP_PATH)
    da = ds_ssta.sst.sel(lon=lon, lat=lat, method="nearest")
    actual_lon = float(da.lon)
    actual_lat = float(da.lat)

    df = pd.DataFrame({
        "anomalies": da.values,
        "year":      da.time.dt.year.values,
        "time":      da.time.values,
    })
    df["continuous_time"] = (da.time.dt.year + (da.time.dt.dayofyear - 1) / 365.25).values
    q95_thr = np.quantile(df["anomalies"].dropna(), q=config.EXTREME_QUANTILE)
    df["q95"] = (df["anomalies"] > q95_thr).astype(int)

    # Trend
    df_clean = df.dropna().copy()
    X = df_clean["continuous_time"].to_numpy().reshape(-1, 1)
    y = df_clean["anomalies"].to_numpy().reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    trend = np.round(reg.coef_[0, 0] * 10, 3)
    df_clean["pred_ols"] = reg.predict(X)[:, 0]
    rolling_avg = da.rolling(time=int(config.FREQ_PER_YEAR_MIN) * config.ROLLING_YEARS, center=True).mean()

    fig_trend = go.Figure([
        go.Scatter(x=df["time"], y=df["anomalies"], mode="lines", name="Anomaly",
                   line=dict(color=theme.BLUE, width=1.5)),
        go.Scatter(x=df_clean["time"], y=df_clean["pred_ols"], mode="lines", name="OLS trend",
                   line=dict(color=theme.YELLOW, width=2.3)),
        go.Scatter(x=da.time.values, y=rolling_avg.values, mode="lines",
                   name=f"{config.ROLLING_YEARS}-year rolling mean",
                   line=dict(color=theme.RED, width=2.5)),
    ])
    fig_trend.update_layout(
        title=f"Anomaly at ({actual_lon:.2f}°, {actual_lat:.2f}°) | Trend: {trend:.3f} °C/decade",
        yaxis_title="SST anomaly (°C)", xaxis_title="Time",
        hovermode='x unified',
        **theme.PLOT_LAYOUT,
    )

    # Extreme events
    df_ext = df.loc[df["q95"] == 1]
    fig_extreme = go.Figure([
        go.Scatter(x=df["time"], y=df["anomalies"], mode="lines", name="Anomaly",
                   line=dict(color=theme.GRAY, width=1), opacity=0.6),
        go.Scatter(x=df_ext["time"], y=df_ext["anomalies"], mode="markers",
                   name="Extreme events", marker=dict(color=theme.RED, size=6, opacity=0.9)),
    ])
    fig_extreme.add_hline(
        y=q95_thr, line_dash="dash", line_color=theme.RED, line_width=2,
        annotation_text=f"q{int(config.EXTREME_QUANTILE * 100)}",
    )
    fig_extreme.update_layout(
        title=f"Extreme events above {int(config.EXTREME_QUANTILE * 100)}th percentile",
        yaxis_title="SST anomaly (°C)", xaxis_title="Time",
        **theme.PLOT_LAYOUT,
    )

    # Bar + KDE
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    full_years = pd.Index(np.arange(year_min, year_max + 1), name="year")
    counts = df.loc[df["q95"] == 1].groupby("year")["q95"].sum().reindex(full_years, fill_value=0)
    df_bar = counts.rename("count").reset_index()

    fig_bar = go.Figure([
        go.Bar(x=df_bar["year"], y=df_bar["count"], name="Extreme events",
               marker_color=theme.RED, opacity=0.5),
    ])
    expanded = np.repeat(df_bar["year"].values, df_bar["count"].values)
    add_kde_overlay(fig_bar, expanded, year_min, year_max, peak=df_bar["count"].max())
    fig_bar.update_layout(
        title="Number of Extreme Events per Year",
        xaxis_title="Year", yaxis_title="Number of extreme events",
        **theme.PLOT_LAYOUT,
    )

    return fig_trend, fig_extreme, fig_bar



# ============================================================================
#  FORECASTING FIGURES
# ============================================================================

def forecast_map_fig(metric_key, lead, metric_options, rmse_max):
    metric_ds = open_zarr(config.FORECAST_ACC_PATH)
    if metric_ds is None:
        return go.Figure()

    if metric_key == "acc_diff":
        da = metric_ds["model_acc"].sel(lead_time=lead) - metric_ds["persistence_acc"].sel(lead_time=lead)
    elif metric_key == "forecast_skill":
        da = 1 - (metric_ds["model_rmse"].sel(lead_time=lead) / metric_ds["persistence_rmse"].sel(lead_time=lead))
    else:
        da = metric_ds[metric_key].sel(lead_time=lead)

    is_diff = "diff" in metric_key
    if "acc" in metric_key:
        colorscale = "RdBu_r" if is_diff else "RdYlBu_r"
        zmin, zmax = (-0.5, 0.5) if is_diff else (-1, 1)
    elif "rmse" in metric_key:
        colorscale = "RdBu_r" if is_diff else "YlOrRd"
        zmin, zmax = (-rmse_max / 2, rmse_max / 2) if is_diff else (0, rmse_max)
    else:
        colorscale = theme.SKILL_COLORSCALE
        zmin, zmax = -5, 1

    label = next(k for k, v in metric_options.items() if v == metric_key)
    return heatmap_fig(
        z=da.values, lons=da.lon.values, lats=da.lat.values,
        colorscale=colorscale,
        title=f"{label}  —  lead time = {lead} days",
        zmin=zmin, zmax=zmax,
    )

def forecast_ts_fig(lon, lat, anchor_date, model_name):
    forecast_ds = open_zarr(config.FORECAST_CHART_PATH)
    if forecast_ds is None:
        return go.Figure()

    anchor_t = pd.Timestamp(anchor_date)
    sel = dict(anchor_time=np.datetime64(anchor_date), lon=lon, lat=lat)
    ctx   = forecast_ds.input_context.sel(**sel, method="nearest").values
    pred_da  = forecast_ds.model_pred.sel(**sel, method="nearest")
    pred = pred_da.values
    truth = forecast_ds.truth.sel(**sel, method="nearest").values
    n_in    = int(forecast_ds.input_window)
    horizon = int(forecast_ds.horizon)

    input_dates    = [anchor_t + pd.Timedelta(days=d) for d in range(-n_in + 1, 1)]
    forecast_dates = [anchor_t + pd.Timedelta(days=d) for d in range(1, horizon + 1)]
    persistence    = np.full(horizon, float(ctx[-1]))

    actual_lon = float(pred_da.lon)
    actual_lat = float(pred_da.lat)

    fig = go.Figure([
        go.Scatter(x=input_dates,    y=ctx,         mode="lines", name="observed (input)",
                   line=dict(color=theme.BLUE,  dash="solid", width=2.2)),
        go.Scatter(x=forecast_dates, y=truth,        mode="lines", name="observed (truth)",
                   line=dict(color=theme.BLUE,  dash="dash",  width=2.2)),
        go.Scatter(x=forecast_dates, y=pred,         mode="lines", name=f"{model_name} forecast",
                   line=dict(color=theme.RED, dash="dash",  width=2.2)),
        go.Scatter(x=forecast_dates, y=persistence,  mode="lines", name="persistence",
                   line=dict(color=theme.YELLOW,   dash="dot",   width=2.2)),
    ])
    fig.add_vline(x=anchor_date, line_dash="dash", line_color=theme.GRAY, line_width=1.2)
    fig.update_layout(
        title=f"Forecast at ({actual_lon:.1f}°, {actual_lat:.1f}°) — start: {anchor_date}",
        xaxis_title="Date", yaxis_title="SSTA (°C)",
        hovermode='x unified',
        **theme.PLOT_LAYOUT,
    )
    return fig


# ============================================================================
#  MARINE HEAT WAVES FIGURES
# ============================================================================

def mhw_map_fig(metric, year):
    ds = open_zarr(config.MHW_MAP_PATH)
    da = ds[metric].sel(year=year)
    return heatmap_fig(
        z=da.values, lons=da.lon.values, lats=da.lat.values,
        colorscale="Hot",
        title=f"MHW {metric.replace('_', ' ')} ({year})",
        zmin=0, zmax=float(np.nanmax(ds[metric].values)),
    )


def mhw_ts_fig(metric, lon, lat):
    ds = open_zarr(config.MHW_MAP_PATH)
    ylabel = "Days per year" if metric == "day_per_year" else "Events per year"
    da = ds[metric].sel(lon=lon, lat=lat, method="nearest")
    actual_lon, actual_lat = float(da.lon), float(da.lat)

    df_ts = pd.DataFrame({
        "year": [int(y) for y in da.year.values],
        ylabel: da.values,
    }).dropna()

    fig = go.Figure([
        go.Bar(x=df_ts["year"], y=df_ts[ylabel], name=ylabel, marker_color=theme.RED, opacity=0.5),
    ])
    expanded = np.repeat(df_ts["year"].values, df_ts[ylabel].values.astype(int).clip(0))
    add_kde_overlay(fig, expanded, df_ts["year"].min(), df_ts["year"].max(), peak=df_ts[ylabel].max())
    fig.update_layout(
        title=f"MHW {ylabel} at ({actual_lon:.2f}°, {actual_lat:.2f}°)",
        xaxis_title="Year", yaxis_title=ylabel,
        **theme.PLOT_LAYOUT,
    )
    return fig