"""The pump-screen pipeline, drawn with Graphviz.

Produces the end-to-end workflow the Day 16 deck links to: raw PDI tables on
the left, the constructed visit label in the middle, the two models, and the
two-stage decision on the right -- then the second item, and the checks that
say any of it works.

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
        node(c, "relevance", "Category Relevance",
             "classification<br/>6 × calibrated binary (GBT)")
        node(c, "affinity", "SKU Affinity",
             "ALS, implicit feedback<br/>item = SKU × daypart")
        node(c, "bestseller", "Best-Seller",
             "popularity fallback<br/>~68% of fills", color=RIDGE,
             fillcolor="#F4F7F9")
        c.body.append("{rank=same; affinity; bestseller}")

    # --- decision ------------------------------------------------------
    node(g, "stage1", "Stage one · the category",
         "highest P(category) we<br/>are <b>allowed</b> to show", color=ORANGE,
         fillcolor="#FEF5E9", penwidth="2")
    node(g, "stage2", "Stage two · the SKU",
         "ranked on <b>P(buys)</b>, inside<br/>that one category only",
         color=ORANGE, fillcolor="#FEF5E9", penwidth="2")
    # The naive P x margin - cost form is wrong: it never compares against
    # the world where no offer was shown. See the Day 16 decision rule.
    node(g, "ev", "Offer Economics",
         "show iff<br/>p₁(margin − discount) &gt; p₀ × margin", color=GOOD,
         fillcolor="#EDF7F2", penwidth="2.2")
    # The anchor is what they buy anyway, so it is the wrong thing to
    # discount. The second item is a different question off a different table.
    node(g, "pairs", "Pair affinity",
         "cohort index &amp; excess<br/>exposure controlled", color=TEAL,
         fillcolor="#EEF5F8", penwidth="1.8")
    node(g, "second", "The second item",
         "the complement a discount<br/>could actually <b>add</b>",
         color=GOOD, fillcolor="#EDF7F2", penwidth="2")
    node(g, "valid", "Validation",
         "calibration by decile<br/>hit@10 vs the best seller", color=GOOD,
         fillcolor="#EDF7F2", penwidth="1.8")
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

    g.edge("ctx", "relevance")
    g.edge("hist", "relevance")
    g.edge("hist", "affinity")
    g.edge("ctx", "bestseller", style="dashed")

    g.edge("relevance", "stage1", label="P(category)")
    g.edge("stage1", "affinity", label="the chosen\\ncategory")
    g.edge("affinity", "stage2", label="personalised")
    g.edge("bestseller", "stage2", label="no history", style="dashed")
    g.edge("stage2", "ev")
    g.edge("ev", "screen")
    g.edge("stage1", "pairs", style="dashed", label="the anchor")
    g.edge("pairs", "second")
    g.edge("second", "screen", style="dashed")
    # Validation hangs off the two STAGES, not off the model objects: what
    # gets checked is the probability we wrote down and the SKU we ranked
    # first, which is also what keeps the edges short.
    g.edge("stage1", "valid", style="dashed", label="calibration",
           color=GOOD, fontcolor=GOOD)
    g.edge("stage2", "valid", style="dashed", label="hit@10",
           color=GOOD, fontcolor=GOOD)
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
