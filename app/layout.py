import path_setup  # noqa: F401  sets up sys.path; import first

import os
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

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

def statCard(icon, label, values, units, descs, source, color_class, border_color):
    return html.Div([
        html.Div([html.I(className=f"fas {icon} fa-2x me-2 {color_class}"), label], className="stat-label"),
        html.Div([html.Span(values[0], className=f"stat-big {color_class}"), html.Span(units[0], className="stat-unit")]),
        html.Div(descs[0], className="stat-desc"),
        html.Hr(className="stat-divider"),
        html.Div([html.Span(values[1], className=f"stat-big {color_class}"), html.Span(units[1], className="stat-unit")]),
        html.Div(descs[1], className="stat-desc"),
        html.Hr(className="stat-divider"),
        html.Div(source, className="stat-source"),
    ], className="overview-stat-card", style={"borderTop": f"3px solid {border_color}"})

def loadingGraphCard(fig_id, height='400px'):
    # loading allows to display a loading widget while the figure is being updated
    # useful when it takes ~5 seconds

    return dcc.Loading(
        html.Div( 
            dcc.Graph(id=fig_id),
            className="chart-card")
    )

def build_layout():
    has_mhw = os.path.exists(config.MHW_MAP_PATH)
    if has_mhw:
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
    
    # ── Tab 0: Overview ──────────────────────────────────────────────────────
    tab_overview = dcc.Tab(label="Overview", value="tab-overview", children=[
        dbc.Container(fluid=True, className="tab-content overview-tab", children=[

            # Hero
            html.Div([
                html.H1(
                    "Global ocean temperatures are changing. Explore 144 years of data.",
                    className="overview-hero-title",
                ),
                html.P(
                    "From weekly SST anomaly animations to machine-learning forecasts. "
                    "Track warming trends, marine heatwaves, and extreme events across the global ocean.",
                    className="overview-hero-sub",
                ),
                html.Div([
                    html.Span([html.I(className="fas fa-calendar-alt me-1"), "1882–2025"],  className="overview-badge"),
                    html.Span([html.I(className="fas fa-globe me-1"),        "Global coverage"], className="overview-badge"),
                    html.Span([html.I(className="fas fa-satellite me-1"),    "Satellite"], className="overview-badge"),
                ], className="overview-badges"),
            ], className="overview-hero"),

            # Stat cards 1×4
            dbc.Row([
                dbc.Col(statCard(
                    "fa-chart-line",
                    "SST Warming Rate",
                    ["+0.88°C", "2025"],
                    [" ", " "],
                    ["since pre-industrial era", "ranked 3rd in NOAA's global temperature record"],
                    "Source: NOAA · IPCC 6th Assessment Report (2021)",
                    "stat-red",    "#e85555"), md=4),
                dbc.Col(statCard(
                    "fa-fire",
                    "Marine Heatwave Days",
                    ["+54%", "90%"],
                    [" ", " "],
                    ["annual MHWD days (1987-2016 vs. 1925-1954)", "of all MHWs linked to human-caused warming"],
                    "Source: IPCC 6th Assessment Report (2021)",
                    "stat-orange", "#e3bb2a"), md=4),
                dbc.Col(statCard(
                    "fa-water",
                    "Ocean Heat",
                    ["90%", "+0.396"],
                    [" ", " Yottajoule"],
                    ["of Earth's excess heat is stored in the oceans", "ocean heat gain (1971-2018)"],
                    "Source: IPCC 6th Assessment Report (2021)",
                    "stat-blue",   "#00b4d8"), md=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(statCard(
                    "fa-person-swimming",
                    "Sea Level",
                    ["3.7 mm/yr", "60-82%"],
                    [" ", " "],
                    ["current rise rate (2006-2018), up 60% since 1971", "of tide gauges will see once-per-century floods annually by 2100"],
                    "Source: NOAA · IPCC 6th Assessment Report (2021)",
                    "stat-teal",   "#2dd4bf"), md=4),
                dbc.Col(statCard(
                    "fa-snowflake",
                    "Artic Level",
                    ["2nd", "2050"],
                    [" lowest on record", " "],
                    ["3.93M miles square ice in 2025", "Arctic Ocean will be sea-ice free in summer"],
                    "Source: NOAA · IPCC 6th Assessment Report (2021)",
                    "stat-pink",   "#c084fc"), md=4),
                dbc.Col(statCard(
                    "fa-tornado",
                    "Tropical Cyclones",
                    ["101", "24"],
                    [" ", " "],
                    ["named storms occurred globally in 2025", "reached major intensity (winds≥111 mph)"],
                    "Source: NOAA",
                    "stat-green",  "#4ade80"), md=4),
            ], className="mb-4"),


            # Explore the dashboard
            html.H5("Explore the dashboard", className="overview-section-title"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.I(className="fas fa-regular fa-video fa-2x overview-nav-icon overview-nav-icon-blue"),
                    html.Strong("SST anomalies", className="overview-nav-title"),
                    html.P("40+ years of weekly anomaly animations.", className="overview-nav-text"),
                    html.Button("Explore →", id="nav-to-video",    n_clicks=0, className="overview-nav-link overview-nav-link-blue"),
                ], className="overview-nav-card")),
                dbc.Col(html.Div([
                    html.I(className="fas fa-regular fa-compass fa-2x overview-nav-icon overview-nav-icon-green"),
                    html.Strong("Anomaly explorer", className="overview-nav-title"),
                    html.P("Click any location for trends and extremes.", className="overview-nav-text"),
                    html.Button("Explore →", id="nav-to-anomaly", n_clicks=0, className="overview-nav-link overview-nav-link-green"),
                ], className="overview-nav-card")),
                dbc.Col(html.Div([
                    html.I(className="fas fa-thermometer-half fa-2x overview-nav-icon overview-nav-icon-orange"),
                    html.Strong("Marine heatwaves", className="overview-nav-title"),
                    html.P("Track MHW frequency year by year.", className="overview-nav-text"),
                    html.Button("Explore →", id="nav-to-mhw",     n_clicks=0, className="overview-nav-link overview-nav-link-orange"),
                ], className="overview-nav-card")),
                dbc.Col(html.Div([
                    html.I(className="fas fa-hexagon-nodes fa-2x overview-nav-icon overview-nav-icon-purple"),
                    html.Strong("SST forecasting", className="overview-nav-title"),
                    html.P("ConvLSTM predictions up to 28 days ahead.", className="overview-nav-text"),
                    html.Button("Explore →", id="nav-to-forecast", n_clicks=0, className="overview-nav-link overview-nav-link-purple"),
                ], className="overview-nav-card")),
            ], className="mb-4"),

            # Footer
            html.Footer(
                "Data: NOAA OISST V2",
                className="overview-footer",
            ),
        ]),
    ])

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
                    loadingGraphCard(fig_id="anomaly-map", height='500px'),
                    textCard("Description", analysis.ANOMALY_CAPTION),
                    textCard("Analysis",analysis.ANOMALY_ANALYSIS),
                ], ),
                dbc.Col([
                        loadingGraphCard(fig_id="anomaly-trend",),
                        loadingGraphCard(fig_id="anomaly-extreme"),
                        loadingGraphCard(fig_id="anomaly-bar"),
                ], ),
            ]),
        ]),
    ])

    # ── Tab 3: MHW ───────────────────────────────────────────────────────────
    if not has_mhw:
        tab_mhw = dcc.Tab(label="Marine HeatWave Visualization", value="tab-mhw", children=[
            dbc.Container(fluid=True, className="tab-content", children=[
                textCard("MHW dataset not found", "Please see the README for instructions on generating this file."),
            ]),
        ])
    else:
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
                        loadingGraphCard(fig_id="mhw-map"),
                        loadingGraphCard(fig_id="mhw-ts"),
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
                        loadingGraphCard(fig_id="forecast-map"),
                        loadingGraphCard(fig_id="forecast-ts"),
                    ], ),
                ]),
            ]),
        ])

    dark_mode_switch =  html.Span([
        dbc.Label(className="fa fa-sun", html_for="switch"),
        dbc.Switch(id="switch-theme", value=True, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-moon", html_for="switch"),
    ])


    return dbc.Container( 
        fluid=True,
        id="page-wrapper",
        children=[ 
            html.Div([
                html.H4("Sea Surface Temperature Explorer Dashboard"),
                dark_mode_switch,],
                className='app-header'
                ),
            dcc.Tabs(
                id="main-tabs", value="tab-overview",
                children=[tab_overview, tab_video, tab_anomaly, tab_mhw, tab_forecast],
            ),
    ])

