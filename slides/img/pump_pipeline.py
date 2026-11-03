"""The pump-screen pipeline, drawn with Graphviz.

Produces the end-to-end workflow the Day 16 deck links to: raw PDI tables on
the left, the constructed visit label in the middle, the three models, and
the expected-value decision on the right.

The deck can only show one stage at a time. This is the whole thing on one
page, which is what you want when someone asks "where does that number come
from?".

Needs the `dot` binary:  brew install graphviz

Run:  uv run --with graphviz python pump_pipeline.py
"""

from pathlib import Path

import graphviz

OUT = Path(__file__).parent / "pump_pipeline"

# The course palette, from _brand.yml.
NAVY = "#022F4A"
RIDGE = "#426A8A"
TEAL = "#4D93AC"
SKY = "#BDD3E5"
CLOUD = "#F4F7F9"
ORANGE = "#F59625"
GOOD = "#1F8A5B"
BAD = "#C7352B"
INK = "#111111"

MONO = "Fira Code, DejaVu Sans Mono, monospace"
SANS = "Open Sans, Helvetica, sans-serif"


def node(g, name, title, sub="", **kw):
    """A two-line box: bold title over muted detail.

    `id` lands in the SVG as-is, which is how pipeline.qmd finds the node
    to attach a click handler to.
    """
    label = f"<<b>{title}</b>"
    if sub:
        label += f'<br/><font point-size="10" color="{RIDGE}">{sub}</font>'
    label += ">"
    kw.setdefault("id", f"n-{name}")
    kw.setdefault("tooltip", "Click for how this looks in our data")
    g.node(name, label=label, **kw)


def build() -> graphviz.Digraph:
    g = graphviz.Digraph("pump_pipeline", format="svg")
    g.attr(
        rankdir="TB",
        bgcolor="white",
        splines="spline",
        nodesep="0.45",
        ranksep="0.5",
        fontname=SANS,
    )
    g.attr("node", shape="box", style="rounded,filled", fontname=SANS,
           fontsize="13", color=SKY, fillcolor=CLOUD, fontcolor=INK,
           margin="0.18,0.12", penwidth="1.4")
    g.attr("edge", color=RIDGE, penwidth="1.4", arrowsize="0.8",
           fontname=SANS, fontsize="10", fontcolor=RIDGE)

    # --- source tables -------------------------------------------------
    with g.subgraph(name="cluster_raw") as c:
        c.attr(label="  PDI tables  ", labeljust="l", style="rounded",
               color=SKY, fontname=MONO, fontsize="11", fontcolor=RIDGE,
               bgcolor="white")
        c.attr("node", shape="box", style="filled", color=SKY,
               fillcolor="white", fontname=MONO, fontsize="11")
        for key, lbl in [
            ("items", "transaction-items-all\\n128.6M lines"),
            ("shopper", "shopper-id\\n8.0M shoppers"),
            ("gtin", "master-gtin\\n99k products"),
            ("stores", "stores-information\\n940 transacting"),
        ]:
            c.node(key, lbl, id=f"n-{key}",
                   tooltip="Click for how this looks in our data")
        c.body.append("{rank=same; items; gtin; shopper; stores}")

    # --- building the unit of analysis ---------------------------------
    node(g, "clean", "Clean lines",
         "qty &gt; 0, amount &gt; 0<br/>coalesce category")
    node(g, "drop", "Drop non-purchases",
         "FUEL is the anchor<br/>SERVICES = card fees", color=BAD,
         fillcolor="#FBEFEE")
    node(g, "visit", "Fuel visit",
         "anchor on fuel<br/>window: −15 / +45 min", color=ORANGE,
         fillcolor="#FEF5E9", penwidth="2.2")
    node(g, "label", "The label",
         "36.1M visits<br/><b>12.8% buy inside</b>", color=ORANGE,
         fillcolor="#FEF5E9", penwidth="2.2")

    # --- features ------------------------------------------------------
    node(g, "ctx", "Context features",
         "hour (store local), weekday<br/>store, pump, grade, gallons")
    node(g, "hist", "History features",
         "per category, strictly<br/>before this visit")

    # --- models --------------------------------------------------------
    with g.subgraph(name="cluster_models") as c:
        c.attr(label="  The models  ", labeljust="l", style="rounded",
               color=TEAL, fontname=MONO, fontsize="11", fontcolor=TEAL,
               bgcolor="white")
        c.attr("node", shape="box", style="rounded,filled", color=TEAL,
               fillcolor="#EEF5F8", fontname=SANS, fontsize="13",
               penwidth="1.8", margin="0.18,0.12")
        node(c, "A", "A · Come inside?",
             "GBTClassifier, binary<br/>context only — calibrated")
        node(c, "B", "B · Which category?",
             "6 × binary GBT<br/>multi-label, not softmax")
        node(c, "C", "C · Which SKU?",
             "ALS, implicit feedback<br/>item = SKU × daypart")
        c.body.append("{rank=same; A; B}")

    # --- decision ------------------------------------------------------
    node(g, "rank", "Re-rank",
         "margin · stock<br/><b>who funds the offer</b>", color=ORANGE,
         fillcolor="#FEF5E9", penwidth="2")
    # The naive P x margin - cost form is wrong: it never compares against
    # the world where no offer was shown. See the Day 16 decision rule.
    node(g, "ev", "Expected value",
         "show iff<br/>p₁(margin − discount) &gt; p₀ × margin", color=GOOD,
         fillcolor="#EDF7F2", penwidth="2.2")
    node(g, "screen", "The pump screen",
         "one offer — or <b>nothing</b>", color=NAVY, fillcolor=NAVY,
         fontcolor="white", penwidth="2")
    node(g, "holdout", "Holdout",
         "no offer shown<br/><b>does not exist yet</b>", color=BAD,
         fillcolor="white", style="rounded,filled,dashed")
    node(g, "did", "Staggered adoption",
         "stores start promos on<br/>different dates — biased", color=BAD,
         fillcolor="white", style="rounded,filled,dashed")

    # --- wiring --------------------------------------------------------
    g.edge("items", "clean")
    g.edge("gtin", "clean", label="GTIN")
    g.edge("shopper", "visit", label="SHOPPER_ID")
    g.edge("stores", "ctx", style="dashed")
    g.edge("clean", "drop")
    g.edge("drop", "visit")
    g.edge("visit", "label")

    g.edge("label", "ctx", style="dashed", arrowhead="none")
    g.edge("label", "hist", style="dashed", arrowhead="none")

    g.body.append("{rank=same; ctx; hist}")

    g.edge("ctx", "A")
    g.edge("ctx", "B", style="dashed")
    g.edge("hist", "B")
    g.edge("hist", "C")

    g.edge("A", "ev", label="P(inside)")
    g.edge("B", "C", label="top category")
    g.edge("C", "rank", label="top 20 SKUs")
    g.edge("rank", "ev")
    g.edge("ev", "screen")
    g.edge("screen", "holdout", style="dashed", label="hold back\\na slice")
    g.edge("holdout", "ev", style="dashed", constraint="false",
           label="measures uplift", color=BAD, fontcolor=BAD)
    g.edge("did", "ev", style="dashed", constraint="false",
           label="second best", color=BAD, fontcolor=BAD)
    g.body.append("{rank=same; holdout; did}")

    return g


def write_partial(svg_path: Path) -> Path:
    """Wrap the SVG in a raw-HTML block Quarto can include.

    An <img> cannot be clicked into, so the markup has to live in the page.
    Width and height are stripped so the viewBox drives sizing.
    """
    svg = svg_path.read_text()
    svg = svg[svg.index("<svg"):]
    for attr in ("width", "height"):
        i = svg.index(f'{attr}="')
        j = svg.index('"', i + len(attr) + 2) + 1
        svg = svg[:i] + svg[j:]
    svg = svg.replace("<svg ", '<svg id="pipeline-svg" ', 1)

    out = svg_path.parent.parent / "_pipeline_diagram.qmd"
    out.write_text("```{=html}\n" + svg + "\n```\n")
    return out


def main() -> None:
    g = build()
    svg = g.render(str(OUT), format="svg", cleanup=True)
    png = g.render(str(OUT), format="png", cleanup=True)
    partial = write_partial(Path(svg))
    print(f"wrote {svg}")
    print(f"wrote {png}")
    print(f"wrote {partial}")


if __name__ == "__main__":
    main()
