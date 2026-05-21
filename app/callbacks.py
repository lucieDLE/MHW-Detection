import plotly.graph_objects as go
from dash import Input, Output, State, callback

import config
import figures
from layout import textCard

# ============================================================================
#  APP CALLBACKS
# ============================================================================

@callback(
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
    return figures.anomaly_figs(lon, lat)


@callback(
    Output("mhw-map", "figure"),
    Input("mhw-metric", "value"),
    Input("mhw-year",   "value"),
)
def update_mhw_map(metric, year):
    return figures.mhw_map_fig(metric, year)


@callback(
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
    return figures.mhw_ts_fig(metric, lon, lat)


@callback(
    Output("forecast-map",      "figure"),
    Output("forecast-analysis", "children"),
    Input("forecast-metric", "value"),
    Input("forecast-lead",   "value"),
    State("forecast-meta",   "data"),
)
def update_forecast_map(metric_key, lead, meta):
    if meta is None:
        return go.Figure(), ""
    fig = figures.forecast_map_fig(metric_key, lead, meta["metric_options"], meta["rmse_max"])
    card = textCard("ANALYSIS", figures.METRIC_TO_ANALYSIS.get(metric_key, ""))
    return fig, card


@callback(
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
    return figures.forecast_ts_fig(lon, lat, anchor_date, meta["model_name"])
