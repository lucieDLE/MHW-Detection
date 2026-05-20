from pathlib import Path
import sys
import functools
import os 
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for p in (str(ROOT_DIR), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import dash_analysis

# ============================================================================
#  COLORS AND FIGURES OPTION
# ============================================================================
HEADER_BG = "#006494"   # deep blue  — top header bar
ACCENT    = "#247ba0"   # ocean blue — active tab, primary highlight


# ── Shared figure options ─────────────────────────────────────────────────────

_MAP_LAYOUT = dict(
    height=config.MAP_HEIGHT,
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
)

_PLOT_LAYOUT = dict(
    height=config.RIGHT_PLOT_HEIGHT,
    margin=dict(l=60, r=20, t=50, b=50),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10)),
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
)

# ============================================================================
#  DATA LOADING
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
        
    fig.update_layout(title=title, **_MAP_LAYOUT, template='simple_white')
    return fig


# ============================================================================
#  ANOMALY FIGURES
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
    rolling_avg = da.rolling(
        time=int(config.FREQ_PER_YEAR_MIN) * config.ROLLING_YEARS, center=True
    ).mean()

    fig_trend = go.Figure([
        go.Scatter(x=df["time"], y=df["anomalies"], mode="lines", name="Anomaly",
                   line=dict(color="#2b8cbe", width=1.5)),
        go.Scatter(x=df_clean["time"], y=df_clean["pred_ols"], mode="lines", name="OLS trend",
                   line=dict(color="#ff8c00", width=2.3)),
        go.Scatter(x=da.time.values, y=rolling_avg.values, mode="lines",
                   name=f"{config.ROLLING_YEARS}-year rolling mean",
                   line=dict(color="#d7301f", width=2.5)),
    ])
    fig_trend.update_layout(
        title=f"Anomaly at ({actual_lon:.2f}°, {actual_lat:.2f}°) | Trend: {trend:.3f} °C/decade",
        yaxis_title="SST anomaly (°C)", xaxis_title="Time",
        **_PLOT_LAYOUT,
    )

    # Extreme events
    df_ext = df.loc[df["q95"] == 1]
    fig_extreme = go.Figure([
        go.Scatter(x=df["time"], y=df["anomalies"], mode="lines", name="Anomaly",
                   line=dict(color="gray", width=1), opacity=0.6),
        go.Scatter(x=df_ext["time"], y=df_ext["anomalies"], mode="markers",
                   name="Extreme events", marker=dict(color="red", size=6, opacity=0.9)),
    ])
    fig_extreme.add_hline(
        y=q95_thr, line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"q{int(config.EXTREME_QUANTILE * 100)}",
    )
    fig_extreme.update_layout(
        title=f"Extreme events above {int(config.EXTREME_QUANTILE * 100)}th percentile",
        yaxis_title="SST anomaly (°C)", xaxis_title="Time",
        **_PLOT_LAYOUT,
    )

    # Bar + KDE
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    full_years = pd.Index(np.arange(year_min, year_max + 1), name="year")
    counts = df.loc[df["q95"] == 1].groupby("year")["q95"].sum().reindex(full_years, fill_value=0)
    df_bar = counts.rename("count").reset_index()

    fig_bar = go.Figure([
        go.Bar(x=df_bar["year"], y=df_bar["count"], name="Extreme events",
               marker_color="#74a9cf", opacity=0.5),
    ])
    expanded = np.repeat(df_bar["year"].values, df_bar["count"].values)
    if expanded.size >= 2:
        kde = gaussian_kde(expanded, bw_method=0.35)
        x_grid = np.linspace(year_min, year_max, 400)
        y_kde = kde(x_grid)
        fig_bar.add_trace(go.Scatter(
            x=x_grid, y=y_kde * (df_bar["count"].max() / y_kde.max()),
            mode="lines", name="KDE", line=dict(color="#74a9cf", width=2.5),
        ))
    fig_bar.update_layout(
        title="Number of Extreme Events per Year",
        xaxis_title="Year", yaxis_title="Number of extreme events",
        **_PLOT_LAYOUT,
    )

    return fig_trend, fig_extreme, fig_bar


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
        go.Bar(x=df_ts["year"], y=df_ts[ylabel], name=ylabel, marker_color="#74a9cf", opacity=0.5),
    ])
    expanded = np.repeat(df_ts["year"].values, df_ts[ylabel].values.astype(int).clip(0))
    if expanded.size >= 2:
        kde = gaussian_kde(expanded, bw_method=0.35)
        x_grid = np.linspace(df_ts["year"].min(), df_ts["year"].max(), 400)
        y_kde = kde(x_grid)
        fig.add_trace(go.Scatter(
            x=x_grid, y=y_kde * (df_ts[ylabel].max() / y_kde.max()),
            mode="lines", name="KDE", line=dict(color="#74a9cf", width=2.5),
        ))
    fig.update_layout(
        title=f"MHW {ylabel} at ({actual_lon:.2f}°, {actual_lat:.2f}°)",
        xaxis_title="Year", yaxis_title=ylabel,
        **_PLOT_LAYOUT,
    )
    return fig


# ============================================================================
#  FORECASTING FIGURES
# ============================================================================

METRIC_TO_ANALYSIS = {
    "model_acc":        dash_analysis.FORECAST_ACC_CAPTION,
    "persistence_acc":  dash_analysis.FORECAST_ACC_CAPTION,
    "acc_diff":         dash_analysis.FORECAST_DIFF_ACC_CAPTION,
    "model_rmse":       dash_analysis.FORECAST_RMSE_CAPTION,
    "persistence_rmse": dash_analysis.FORECAST_RMSE_CAPTION,
    "forecast_skill":   dash_analysis.FORECAST_SKILL_CAPTION,
}

# Asymmetric diverging colorscale for skill score (range -5 to 1, midpoint at 0)
SKILL_COLORSCALE = [
    [0.000, "rgb(8,48,107)"],
    [0.700, "rgb(158,202,225)"],
    [0.833, "rgb(255,255,255)"],
    [0.917, "rgb(252,146,114)"],
    [1.000, "rgb(165,15,21)"],
]

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
        colorscale = SKILL_COLORSCALE
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
    pred  = forecast_ds.model_pred.sel(**sel, method="nearest").values
    truth = forecast_ds.truth.sel(**sel, method="nearest").values
    n_in    = int(forecast_ds.input_window)
    horizon = int(forecast_ds.horizon)

    input_dates    = [anchor_t + pd.Timedelta(days=d) for d in range(-n_in + 1, 1)]
    forecast_dates = [anchor_t + pd.Timedelta(days=d) for d in range(1, horizon + 1)]
    persistence    = np.full(horizon, float(ctx[-1]))

    actual_lon = float(forecast_ds.model_pred.sel(**sel, method="nearest").lon)
    actual_lat = float(forecast_ds.model_pred.sel(**sel, method="nearest").lat)

    fig = go.Figure([
        go.Scatter(x=input_dates,    y=ctx,         mode="lines", name="observed (input)",
                   line=dict(color="#2b8cbe", dash="solid", width=2.2)),
        go.Scatter(x=forecast_dates, y=truth,        mode="lines", name="observed (truth)",
                   line=dict(color="#2b8cbe", dash="dash",  width=2.2)),
        go.Scatter(x=forecast_dates, y=pred,         mode="lines", name=f"{model_name} forecast",
                   line=dict(color="#d7301f", dash="dash",  width=2.2)),
        go.Scatter(x=forecast_dates, y=persistence,  mode="lines", name="persistence",
                   line=dict(color="#ff8c00", dash="dot",   width=2.2)),
    ])
    fig.add_vline(x=anchor_date, line_dash="dash", line_color="gray", line_width=1.2)
    fig.update_layout(
        title=f"Forecast at ({actual_lon:.1f}°, {actual_lat:.1f}°) — start: {anchor_date}",
        xaxis_title="Date", yaxis_title="SSTA (°C)",
        **_PLOT_LAYOUT,
    )
    return fig


# ============================================================================
#  APP LAYOUT
# ============================================================================

def build_layout():
    ds_mhw = open_zarr(config.MHW_MAP_PATH)
    mhw_years = [int(y) for y in ds_mhw.year.values]
    mhw_marks = {y: (str(y) if y % 5 == 0 else "") for y in mhw_years}

    has_forecast = False
    if os.path.exists(config.FORECAST_ACC_PATH):
        has_forecast = True
        metric_ds = open_zarr(config.FORECAST_ACC_PATH)

        forecast_ds    = open_zarr(config.FORECAST_CHART_PATH)
        model_name     = str(metric_ds.model)
        input_window   = str(metric_ds.input_window)
        rmse_max       = float(metric_ds["rmse_range"].values[1]) if "rmse_range" in metric_ds else 2.0
        lead_times     = [int(v) for v in np.array(config.LEAD_TIMES)]
        anchor_dates   = [str(t)[:10] for t in forecast_ds.anchor_time.values]
        metric_options = {
            f"{model_name} ACC": "model_acc",
            "Persistence ACC":   "persistence_acc",
            "ACC Difference":    "acc_diff",
            f"{model_name} RMSE":"model_rmse",
            "Persistence RMSE":  "persistence_rmse",
            "Forecasting skill": "forecast_skill",
        }
        lead_marks = {v: str(v) for v in lead_times}
    
    # ── Tab 1: Video ──────────────────────────────────────────────────────────
    tab_video = dcc.Tab(label="SST Anomalies (Video)", value="tab-video", children=[
        dbc.Container(fluid=True, className="tab-content", children=[
            dbc.Card(dbc.CardBody(dcc.Markdown(dash_analysis.TIMESERIE_CAPTION))),
            html.Video(
                src="/assets/videos/sst_weekly_combined.mp4",
                controls=True, loop=True,
                style={"width": "100%", "maxHeight": f"{config.TIME_SERIE_HEIGHT}px"},
            ),
        ]),
    ])

    # ── Tab 2: Anomaly Explorer ───────────────────────────────────────────────
    tab_anomaly = dcc.Tab(label="Anomaly Explorer", value="tab-anomaly", children=[
        dbc.Container(fluid=True, className="tab-content", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="anomaly-map", figure=initial_map_fig()),
                    dbc.Card([
                        dbc.CardHeader("Description"),
                        dbc.CardBody(dcc.Markdown(dash_analysis.ANOMALY_CAPTION)),
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Analysis"),
                        dbc.CardBody(dcc.Markdown(dash_analysis.ANOMALY_ANALYSIS)),
                    ]),
                ], width=7),
                dbc.Col([
                    dcc.Loading(dcc.Graph(id="anomaly-trend")),
                    dcc.Loading(dcc.Graph(id="anomaly-extreme")),
                    dcc.Loading(dcc.Graph(id="anomaly-bar")),
                ], width=5),
            ]),
        ]),
    ])

    # ── Tab 3: MHW ───────────────────────────────────────────────────────────
    tab_mhw = dcc.Tab(label="Marine HeatWave Visualization", value="tab-mhw", children=[
        dbc.Container(fluid=True, className="tab-content", children=[
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="mhw-metric",
                        options=[
                            {"label": "Days per year",   "value": "day_per_year"},
                            {"label": "Events per year", "value": "event_per_year"},
                        ],
                        value="day_per_year", clearable=False,
                        className="mb-2",
                    ),
                    html.Label("Year"),
                    dcc.Slider(
                        id="mhw-year",
                        min=mhw_years[0], max=mhw_years[-1],
                        step=None, marks=mhw_marks, value=mhw_years[0],
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dbc.Card([
                        dbc.CardHeader("Description"),
                        dbc.CardBody(dcc.Markdown(dash_analysis.MHW_CAPTION)),
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Analysis"),
                        dbc.CardBody(dcc.Markdown(dash_analysis.MHW_ANALYSIS)),
                    ]),
                ], width=5),
                dbc.Col([
                    dcc.Loading(dcc.Graph(id="mhw-map")),
                    dcc.Loading(dcc.Graph(id="mhw-ts")),
                ], width=7),
            ]),
        ]),
    ])

    # ── Tab 4: Forecast ───────────────────────────────────────────────────────
    if not has_forecast:
        tab_forecast = dcc.Tab(label="SST Forecasting", value="tab-forecast", children=[
            dbc.Container(fluid=True, className="tab-content", children=[
                dbc.Card([
                    dbc.CardHeader("Forecast dataset not found"),
                    dbc.CardBody(dcc.Markdown(
                        "Please see the README for instructions on generating this file."
                    )),
                ]),
            ]),
        ])
    else:
        tab_forecast = dcc.Tab(label="SST Forecasting", value="tab-forecast", children=[
            dcc.Store(id="forecast-meta", data={
                "model_name":     model_name,
                "input_window":   input_window,
                "rmse_max":       rmse_max,
                "metric_options": metric_options,
            }),
            dbc.Container(fluid=True, className="tab-content", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown(
                            f"**Input window:** {input_window} days  |  "
                            f"**Test period:** {config.DL_TEST_RANGE}",
                            className="info-text",
                        ),
                        dcc.Dropdown(
                            id="forecast-metric",
                            options=[{"label": k, "value": v} for k, v in metric_options.items()],
                            value="model_acc", clearable=False,
                            className="mb-2",
                        ),
                        html.Label("Lead time (days)"),
                        dcc.Slider(
                            id="forecast-lead",
                            min=lead_times[0], max=lead_times[-1],
                            step=None, marks=lead_marks, value=lead_times[0],
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Label("Forecast start date"),
                        dcc.Dropdown(
                            id="forecast-anchor",
                            options=[{"label": d, "value": d} for d in anchor_dates],
                            value=anchor_dates[len(anchor_dates) // 2],
                            clearable=False,
                            className="mb-2",
                        ),
                        dbc.Card([
                            dbc.CardHeader("Metric choices"),
                            dbc.CardBody(dcc.Markdown(dash_analysis.FORECAST_METRIC_CAPTION)),
                        ]),
                        html.Div(id="forecast-analysis"),
                    ], width=5),
                    dbc.Col([
                        dcc.Loading(dcc.Graph(id="forecast-map")),
                        dcc.Loading(dcc.Graph(id="forecast-ts")),
                    ], width=7),
                ]),
            ]),
        ])

    return html.Div([
        html.Div("Sea Surface Temperature Explorer Dashboard", className="app-header"),
        dcc.Tabs(
            id="main-tabs", value="tab-video",
            children=[tab_video, tab_anomaly, tab_mhw, tab_forecast],
            colors={"border": "#cfd8dc", "primary": ACCENT, "background": "#eef3f5"},
        ),
    ])


# ── App init ──────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    assets_folder=str(ROOT_DIR / "assets"),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server
app.layout = build_layout()


# ============================================================================
#  APP CALLBACKS
# ============================================================================

@app.callback(
    Output("anomaly-trend",   "figure"),
    Output("anomaly-extreme", "figure"),
    Output("anomaly-bar",     "figure"),
    Input("anomaly-map", "clickData"),
)
def update_anomaly(click_data):
    if click_data is None:
        lon, lat = config.DEFAULT_TAP_LON, config.DEFAULT_TAP_LAT
    else:
        lon = click_data["points"][0]["x"]
        lat = click_data["points"][0]["y"]
    return anomaly_figs(lon, lat)


@app.callback(
    Output("mhw-map", "figure"),
    Input("mhw-metric", "value"),
    Input("mhw-year",   "value"),
)
def update_mhw_map(metric, year):
    return mhw_map_fig(metric, year)


@app.callback(
    Output("mhw-ts", "figure"),
    Input("mhw-map",    "clickData"),
    Input("mhw-metric", "value"),
)
def update_mhw_ts(click_data, metric):
    if click_data is None:
        lon, lat = config.DEFAULT_TAP_LON, config.DEFAULT_TAP_LAT
    else:
        lon = click_data["points"][0]["x"]
        lat = click_data["points"][0]["y"]
    return mhw_ts_fig(metric, lon, lat)


@app.callback(
    Output("forecast-map",      "figure"),
    Output("forecast-analysis", "children"),
    Input("forecast-metric", "value"),
    Input("forecast-lead",   "value"),
    State("forecast-meta",   "data"),
)
def update_forecast_map(metric_key, lead, meta):
    if meta is None:
        return go.Figure(), ""
    fig = forecast_map_fig(metric_key, lead, meta["metric_options"], meta["rmse_max"])
    card = dbc.Card(dbc.CardBody(dcc.Markdown(METRIC_TO_ANALYSIS.get(metric_key, ""))))
    return fig, card


@app.callback(
    Output("forecast-ts", "figure"),
    Input("forecast-map",    "clickData"),
    Input("forecast-anchor", "value"),
    State("forecast-meta",   "data"),
)
def update_forecast_ts(click_data, anchor_date, meta):
    if meta is None or anchor_date is None:
        return go.Figure()
    if click_data is None:
        lon, lat = config.DEFAULT_TAP_LON, config.DEFAULT_TAP_LAT
    else:
        lon = click_data["points"][0]["x"]
        lat = click_data["points"][0]["y"]
    return forecast_ts_fig(lon, lat, anchor_date, meta["model_name"])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
