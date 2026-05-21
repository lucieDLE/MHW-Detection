from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for p in (str(ROOT_DIR), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


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
)
server = app.server
app.layout = build_layout()

if __name__ == "__main__":
    app.run(debug=True, port=8050)
