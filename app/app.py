from path_setup import ROOT_DIR  # sets up sys.path; import first

import dash_bootstrap_components as dbc
from dash import Dash
from layout import build_layout
import callbacks  # registers all @app.callback decorators

# ── App init ──────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    assets_folder=str(ROOT_DIR / "assets"),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.layout = build_layout()

if __name__ == "__main__":
    app.run(debug=True, port=8050)
