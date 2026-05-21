from pathlib import Path
import sys
import os 
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for p in (str(ROOT_DIR), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import analysis
import theme
import figures

# ============================================================================
#  LAYOUT FUNCTIONS
# ============================================================================

def textCard(title="TITLE", text='some text'):
    return html.Div(
        dbc.Card([
            dbc.CardHeader(title),
            dbc.CardBody(dcc.Markdown(text)),
        ]),
    )

def loadingGraphCard(fig_id, figure, height='400px'):
    # loading allows to display a loading widget while the figure is being updated
    # useful when it takes ~5 seconds
    if figure : 
        return dcc.Loading(
            html.Div(
                dcc.Graph(id=fig_id, figure=figure),
                className="chart-card"
            ) )
    else: 
        return dcc.Loading(
            html.Div( 
                dcc.Graph(id=fig_id),
                className="chart-card")
        )

def build_layout():
    ds_mhw = figures.open_zarr(config.MHW_MAP_PATH)
    mhw_years = [int(y) for y in ds_mhw.year.values]
    mhw_marks = {y: (str(y) if y % 5 == 0 else "") for y in mhw_years}

    has_forecast = False
    if os.path.exists(config.FORECAST_ACC_PATH):
        has_forecast = True
        metric_ds = figures.open_zarr(config.FORECAST_ACC_PATH)

        forecast_ds    = figures.open_zarr(config.FORECAST_CHART_PATH)
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
            textCard("Video Description", analysis.TIMESERIE_CAPTION),
            html.Video(
                src="/assets/videos/sst_weekly_combined.mp4",
                controls=True, loop=True,
                style={"width": "100%", "maxHeight": f"{theme.TIME_SERIE_HEIGHT}px"},
            ),
        ]),
    ])

    # ── Tab 2: Anomaly Explorer ───────────────────────────────────────────────
    tab_anomaly = dcc.Tab(label="Anomaly Explorer", value="tab-anomaly", children=[
        dbc.Container(fluid=True, className="tab-content", children=[
            dbc.Row([
                dbc.Col([
                    loadingGraphCard(fig_id="anomaly-map", figure=figures.initial_map_fig(), height='500px'),
                    textCard("Description", analysis.ANOMALY_CAPTION),
                    textCard("Analysis",analysis.ANOMALY_ANALYSIS),
                ], ),
                dbc.Col([
                        loadingGraphCard(fig_id="anomaly-trend",figure=None,),
                        loadingGraphCard(fig_id="anomaly-extreme",figure=None),
                        loadingGraphCard(fig_id="anomaly-bar",figure=None),
                ], ),
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
                    textCard("Description", analysis.MHW_CAPTION),
                    textCard("Analysis", analysis.MHW_ANALYSIS),
                ], ),
                dbc.Col([
                    loadingGraphCard(fig_id="mhw-map",figure=None),
                    loadingGraphCard(fig_id="mhw-ts",figure=None),
                ], ),
            ]),
        ]),
    ])

    # ── Tab 4: Forecast ───────────────────────────────────────────────────────
    if not has_forecast:
        tab_forecast = dcc.Tab(label="SST Forecasting", value="tab-forecast", children=[
            dbc.Container(fluid=True, className="tab-content", children=[
                textCard("Forecast dataset not found", "Please see the README for instructions on generating this file."),
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
                            step=None, marks=lead_marks, value=lead_times[-2],
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
                        textCard("Metric choices", analysis.FORECAST_METRIC_CAPTION),
                        html.Div(id="forecast-analysis"),
                    ], ),
                    dbc.Col([
                        loadingGraphCard(fig_id="forecast-map",figure=None),
                        loadingGraphCard(fig_id="forecast-ts",figure=None),
                    ], ),
                ]),
            ]),
        ])

    return html.Div([
        html.Div("Sea Surface Temperature Explorer Dashboard", className="app-header"),
        dcc.Tabs(
            id="main-tabs", value="tab-video",
            children=[tab_video, tab_anomaly, tab_mhw, tab_forecast],
        ),
    ])

