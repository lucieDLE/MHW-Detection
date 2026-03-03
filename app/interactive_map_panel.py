from pathlib import Path
import sys
from scipy.stats import gaussian_kde

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr
from sklearn.linear_model import LinearRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



pn.extension()
hv.extension("bokeh")
xr.set_options(display_style="text")


def _resolve_data_path(path_from_config):
    # config paths are relative to src/, so resolve from that base.
    return (SRC_DIR / path_from_config).resolve()


def load_analysis_data():
    data_path = _resolve_data_path(config.DATA_PATH)
    ref_path = _resolve_data_path(config.DATA_REF)

    ds = xr.open_dataset(data_path, engine="netcdf4")
    ds_ref = xr.open_dataset(ref_path, engine="netcdf4")

    ds = remove_countries_data(ds, ds_ref)
    ds["time"] = xr.decode_cf(ds).time

    sst_med_sea = ds.sst
    sst_grouped = sst_med_sea.groupby("time.month")
    tos_clim = sst_grouped.mean(dim="time")
    tos_std = sst_grouped.std(dim="time")

    initial_map = tos_std.mean(dim="month")
    tos_anom = sst_grouped - tos_clim

    return initial_map, tos_anom


def compute_anomaly_plot(df: pd.DataFrame, tos_anom_selected: xr.DataArray):
    rolling_year_num = config.ROLLING_YEARS
    sme_step = config.FREQ_PER_YEAR_MIN

    anom_resampled = tos_anom_selected.resample(time="SME").mean("time")
    rolling_year_avg = anom_resampled.rolling(
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
            "rolled_time": anom_resampled.time.values,
            "anomalies_rolled_avg": rolling_year_avg.values,
        }
    )

    anomaly_time_plot = tos_anom_selected.hvplot.line(
        x="time",
        y="sst",
        title=(
            f"Anomaly at selected location "
            f"({float(tos_anom_selected.lat.values):.2f}, {float(tos_anom_selected.lon.values):.2f}) "
            f"| Trend: {trend} ˚C/decade"
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
        shared_axes=False, legend_position="top_right", show_grid=True
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
        title=f"Extreme events above {int(quantile_val * 100)}th quantile threshold",
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
        active_tools=["pan"], legend_position="top_right", legend_cols=3, show_grid=True
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
        return (bar_plot * kde_curve).opts(show_grid=True,legend_position="top_right")

    return bar_plot

def build_app():
    initial_map, tos_anom = load_analysis_data()

    initial_plot = initial_map.hvplot(
        x="lon",
        y="lat",
        cmap="coolwarm",
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
        tos_anom_selected = tos_anom.sel(lon=x, lat=y, method="nearest")

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

        df = df.loc[(df["year"]>= config.MIN_YEAR ) & (df["year"]<= config.MAX_YEAR)]        

        q95 = np.quantile(df["anomalies"], q=config.EXTREME_QUANTILE)
        df["q95"] = (df["anomalies"] > q95).astype(int)

        anomaly_time_plot = compute_anomaly_plot(df, tos_anom_selected)
        extreme_event_plot = compute_extreme_events_plot(df)
        bar_plot = compute_barplot(df)

        return pn.Column(
            anomaly_time_plot,
            extreme_event_plot,
            bar_plot,
            sizing_mode="stretch_width",
            width=config.RIGHT_PANEL_WIDTH,
        )

    right_panel = pn.bind(select_point, x=posxy.param.x, y=posxy.param.y)
    dashboard = pn.Row(
        pn.Column(initial_plot, sizing_mode="stretch_width"),
        pn.Spacer(width=12),
        right_panel,
        sizing_mode="stretch_width",
    )

    return pn.template.FastListTemplate(
        title="Mediterranean SST Anomaly Explorer",
        main=[dashboard],
        accent_base_color="#0d6e6e",
        header_background="#0d6e6e",
    )


app = build_app()
app.servable()


if __name__ == "__main__":
    pn.serve(app, show=True, title="Mediterranean SST Anomaly Explorer")
