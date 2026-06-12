# ── Display dimensions ────────────────────────────────────────────────────────
TIME_SERIE_HEIGHT = 760

# ── Chart color palette ──────────────────────────────────────────────────────
# valurd based on the --cat  assets/custom.css so charts and UI have same colors.
OBSERVED = "#0096c7"        # observed / input / main series        (= --cat-ocean)
TREND    = "#e9a23b"        # OLS linear trend / smoothed-trend KDE (= --cat-amber)
ROLLING  = "#e5604d"        # multi-year rolling mean (warming line)(= --cat-warm)
FORECAST = "#e5604d"        # model forecast line                   (= --cat-warm)
EXTREME  = "#e5604d"        # extreme events / bars / threshold     (= --cat-warm)
BASELINE = "#5c7a8a"        # persistence / secondary / background lines

# ── COLOR PALETTE ───────────────────────────────────────────────────────

# Plot interior (plot_bgcolor) — kept light in both themes so a single gray
# gridcolor stays visible. The page (paper_bgcolor) is what switches dark/light.
dark_inside_plot = "#ededed"
light_inside_plot = "#ffffff"

dark_outer_plot = "#041c30"
light_outer_plot = "#f5fbff"

# ── Colorscales ───────────────────────────────────────────────────────────────
# Asymmetric diverging scale for skill score (range -5 to 1, midpoint at 0)
SKILL_COLORSCALE = [
    [0.000, "rgb(8,48,107)"],
    [0.700, "rgb(158,202,225)"],
    [0.833, "rgb(255,255,255)"],
    [0.917, "rgb(252,146,114)"],
    [1.000, "rgb(165,15,21)"],
]

# ── Shared Plotly figure layouts ──────────────────────────────────────────────
MAP_LAYOUT = dict(
    autosize=True,
    margin=dict(l=20, r=10, t=50, b=20),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    xaxis=dict(showgrid=True, zeroline=False, gridcolor='gray', griddash='dash', minor_griddash="dot"),
    yaxis=dict(showgrid=True, zeroline=False, gridcolor='gray', griddash='dash', minor_griddash="dot"),
)

PLOT_LAYOUT = dict(
    autosize=True,
    margin=dict(l=60, r=20, t=70, b=50),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, zeroline=False, gridcolor='gray', griddash='dash'),
    yaxis=dict(showgrid=True, zeroline=False, gridcolor='gray', griddash='dash'),
)

def apply_theme(fig, dark: bool):
    """Single owner of the figure template and background colors.

    Figure builders set only structure (traces, titles, margins, grids); the
    template and plot/paper backgrounds are applied here so they stay
    consistent across every figure and don't get set in two places.
    """
    if dark:
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=dark_inside_plot,
            paper_bgcolor=dark_outer_plot,
        )
    else:
        fig.update_layout(
            template="ggplot2",
            plot_bgcolor=light_inside_plot,
            paper_bgcolor=light_outer_plot,
        )
    return fig