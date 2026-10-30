"""Default styles for plots."""

from plotly.express.colors import diverging

COLORS = {
    "x=y": "#000000",
    "points": "#377eb8",
    "predictions": "rgba(200, 30, 50, 0.6)",
    "target": "rgba(30, 200, 50, 0.6)",
    "pr_curve": "#377eb8",
    "pr_auc": "rgba(255, 200, 100, 0.4)",
    "roc_curve": "#377eb8",
    "roc_auc": "rgba(255, 200, 100, 0.4)",
}
COLORSCALES = {
    "classes": diverging.Portland,
}

DEFAULT_STYLE = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "width": 1200,
    "height": 800,
    "margin": {"l": 100, "r": 100, "t": 100, "b": 100},
    "xaxis_showgrid": True,
    "yaxis_showgrid": True,
    "xaxis_gridcolor": "darkgrey",
    "yaxis_gridcolor": "darkgrey",
    "xaxis_griddash": "dash",
    "yaxis_griddash": "dash",
    "xaxis_linecolor": "black",
    "yaxis_linecolor": "black",
    "xaxis_nticks": 10,
    "yaxis_nticks": 10,
    "xaxis_showline": True,
    "yaxis_showline": True,
    "xaxis_zeroline": False,
    "yaxis_zeroline": False,
    "yaxis_scaleanchor": "x",
    "yaxis_scaleratio": 1,
    "xaxis_title_font_size": 20,
    "yaxis_title_font_size": 20,
    "xaxis_tickfont_size": 17,
    "yaxis_tickfont_size": 17,
    "legend_font_size": 18,
    "title_font_size": 26,
    "showlegend": True,
}

DEFAULT_COLORSCALE = [
    "#E07585",
    "#EE8975",
    "#FFBF70",
    "#DDD555",
    "#90DA9A",
    "#50D0AA",
    "#6ABFE8",
    "#50B0EF",
    "#8483EB",
    "#C088E0",
]

EQUAL_RATIO = {
    "yaxis": {
        "scaleanchor": "x",
        "scaleratio": 1,
        "showgrid": True,
    },
}

BOXPLOT_STYLE = {}

HISTOGRAM_STYLE = {
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "width": 1200,
    "height": 800,
    "margin": {"l": 100, "r": 100, "t": 100, "b": 100},
    "xaxis_showgrid": True,
    "yaxis_showgrid": True,
    "xaxis_gridcolor": "darkgrey",
    "yaxis_gridcolor": "darkgrey",
    "xaxis_griddash": "dash",
    "yaxis_griddash": "dash",
    "xaxis_linecolor": "black",
    "yaxis_linecolor": "black",
    "xaxis_title_font_size": 20,
    "yaxis_title_font_size": 20,
    "xaxis_tickfont_size": 17,
    "yaxis_tickfont_size": 17,
    "legend_font_size": 18,
    "title_font_size": 26,
    "showlegend": True,
}
