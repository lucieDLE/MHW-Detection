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
    height=MAP_HEIGHT,
    margin=dict(l=60, r=20, t=50, b=50),
    xaxis_title="Longitude",
    yaxis_title="Latitude",
)

PLOT_LAYOUT = dict(
    height=RIGHT_PLOT_HEIGHT,
    margin=dict(l=60, r=20, t=70, b=50),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
)