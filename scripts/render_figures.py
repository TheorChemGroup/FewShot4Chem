"""Render Figures 1, 2 and S2 from their Plotly specifications in `figures/`.

`figures/figure1.plotly.json` and `figures/figure2.plotly.json` hold the published figures
(data + styling). Figure S2 is Figure 2 with the full, unclipped y-axis and is derived here.
`scripts/check_figures.py` verifies every plotted value against `results/`.

    python scripts/render_figures.py            # writes figures/*.png (2x) and *.pdf (vector)

Needs plotly 6.9.0 and kaleido 1.3.0 (the versions the published figures were rendered with; kaleido
needs a Chrome/Chromium, which `plotly_get_chrome` installs). Text is set in Nimbus Sans (a Helvetica clone); install it
(`fonts-urw-base35` on Debian/Ubuntu) or the renderer substitutes another sans-serif face.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
Y_AXES = ("yaxis", "yaxis2", "yaxis3")


def figure_s2(figure2):
    """Figure 2 with every y-axis spanning the full data range.

    The below-axis markers of Figure 2 exist only because of its clipped axis and are dropped.
    The y-title subscript is set to 15 px so it prints at >= 7 pt at 18 cm width.
    """
    fig = deepcopy(figure2)
    fig["data"] = [t for t in fig["data"] if t.get("meta", {}).get("role") != "boundary_indicator"]
    values = [y for t in fig["data"] for y in t.get("y", []) if isinstance(y, (int, float)) and math.isfinite(y)]
    lo, hi = min(values), max(values)
    pad = max(0.04, 0.04 * (hi - lo))
    for axis in Y_AXES:
        fig["layout"][axis]["range"] = [min(lo - pad, 0.0), max(hi + pad, 1.03)]
        for k in ("tickmode", "tickvals", "ticktext"):
            fig["layout"][axis].pop(k, None)
    fig["layout"]["yaxis"]["title"]["text"] = (
        'Performance on ε<sub><span style="font-size:15px">LUMO</span></sub> (R²)'
    )
    return fig


def render(spec, stem):
    import plotly.graph_objects as go
    import plotly.io as pio

    fig = go.Figure(spec)
    w, h = spec["layout"]["width"], spec["layout"]["height"]
    pio.write_image(fig, FIGURES / f"{stem}.png", format="png", scale=2, width=w, height=h)
    pio.write_image(fig, FIGURES / f"{stem}.pdf", format="pdf", scale=1, width=w, height=h)
    print(f"figures/{stem}.png, figures/{stem}.pdf")


@click.command()
@click.option("--no-render", is_flag=True, help="Only (re)write figureS2.plotly.json.")
def main(no_render):
    figure1 = json.loads((FIGURES / "figure1.plotly.json").read_text())
    figure2 = json.loads((FIGURES / "figure2.plotly.json").read_text())
    s2 = figure_s2(figure2)
    (FIGURES / "figureS2.plotly.json").write_text(json.dumps(s2, indent=2, allow_nan=False) + "\n")
    if no_render:
        return
    for spec, stem in ((figure1, "figure1"), (figure2, "figure2"), (s2, "figureS2")):
        render(spec, stem)


if __name__ == "__main__":
    main()
