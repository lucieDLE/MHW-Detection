import plotly.graph_objects as go
from dash import Input, Output, State, callback

import config
import figures
from layout import textCard
import theme
# ============================================================================
#  APP CALLBACKS
# ============================================================================
@callback(Output("page-wrapper", "className"), Input("switch-theme", "value"))
def change_theme(value):
    return "dark" if value else ""

@callback(
    Output("anomaly-map", "figure"),
    Input("switch-theme", "value"),
)
def update_anomaly_map(dark):
    return theme.apply_theme(figures.initial_map_fig(), dark)

@callback(
    Output("anomaly-trend",   "figure"),
    Output("anomaly-extreme", "figure"),
    Output("anomaly-bar",     "figure"),
    Input("anomaly-map", "clickData"),
    Input("switch-theme", "value"),
)
def update_anomaly(click_data, dark):
    if click_data is None:
        lon, lat = config.DEFAULT_TAP_LON, config.DEFAULT_TAP_LAT
    else:
        lon = click_data["points"][0]["x"]
        lat = click_data["points"][0]["y"]
    trend, extreme, bar = figures.anomaly_figs(lon, lat)
    return theme.apply_theme(trend, dark), theme.apply_theme(extreme, dark), theme.apply_theme(bar, dark)


@callback(
    Output("mhw-map", "figure"),
    Input("mhw-metric", "value"),
    Input("mhw-year",   "value"),
    Input("switch-theme", "value"),
)
def update_mhw_map(metric, year, dark):
    return theme.apply_theme(figures.mhw_map_fig(metric, year), dark)


@callback(
    Output("mhw-ts", "figure"),
    Input("mhw-map",    "clickData"),
    Input("mhw-metric", "value"),
    Input("switch-theme", "value"),
)
def update_mhw_ts(click_data, metric, dark):
    if click_data is None:
        lon, lat = config.DEFAULT_TAP_LON, config.DEFAULT_TAP_LAT
    else:
        lon = click_data["points"][0]["x"]
        lat = click_data["points"][0]["y"]
    return theme.apply_theme(figures.mhw_ts_fig(metric, lon, lat),dark)


@callback(
    Output("forecast-map",      "figure"),
    Output("forecast-analysis", "children"),
    Input("forecast-metric", "value"),
    Input("forecast-lead",   "value"),
    Input("switch-theme", "value"),
    State("forecast-meta",   "data"),
)
def update_forecast_map(metric_key, lead, dark, meta):
    if meta is None:
        return go.Figure(), ""
    fig = figures.forecast_map_fig(metric_key, lead, meta["metric_options"], meta["rmse_max"])
    card = textCard("ANALYSIS", figures.METRIC_TO_ANALYSIS.get(metric_key, ""))
    return theme.apply_theme(fig, dark), card


@callback(
    Output("forecast-ts", "figure"),
    Input("forecast-map",    "clickData"),
    Input("forecast-anchor", "value"),
    Input("switch-theme", "value"),
    State("forecast-meta",   "data"),
)
def update_forecast_ts(click_data, anchor_date, dark, meta):
    if meta is None or anchor_date is None:
        return go.Figure()
    if click_data is None:
        lon, lat = config.DEFAULT_TAP_LON, config.DEFAULT_TAP_LAT
    else:
        lon = click_data["points"][0]["x"]
        lat = click_data["points"][0]["y"]
    return theme.apply_theme(figures.forecast_ts_fig(lon, lat, anchor_date, meta["model_name"]), dark)
