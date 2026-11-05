"""Table lineage for the pump-screen model, drawn with Graphviz.

Where `pump_pipeline.py` draws the *reasoning* — label, models, decision —
this one draws the *tables*. Every box is something a team has to actually
materialise, with a grain, a schema and a job that writes it.

Needs the `dot` binary:  brew install graphviz

Run:  uv run --with graphviz python pump_tables.py
"""

from pathlib import Path

import graphviz

OUT = Path(__file__).parent / "pump_tables"

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
    """A box labelled with a table name over its grain."""
    label = f'<<font face="{MONO}"><b>{title}</b></font>'
    if sub:
        label += f'<br/><font point-size="10" color="{RIDGE}">{sub}</font>'
    label += ">"
    kw.setdefault("id", f"n-{name}")
    kw.setdefault("tooltip", "Click for the schema and the code")
    g.node(name, label=label, **kw)


def build() -> graphviz.Digraph:
    g = graphviz.Digraph("pump_tables", format="svg")
    g.attr(rankdir="TB", bgcolor="white", splines="spline",
           nodesep="0.45", ranksep="0.55", fontname=SANS)
    g.attr("node", shape="box", style="rounded,filled", fontname=SANS,
           fontsize="13", color=SKY, fillcolor=CLOUD, fontcolor=INK,
           margin="0.2,0.13", penwidth="1.4")
    g.attr("edge", color=RIDGE, penwidth="1.4", arrowsize="0.8",
           fontname=SANS, fontsize="10", fontcolor=RIDGE)

    # ---- bronze: what PDI delivered -------------------------------
    with g.subgraph(name="cluster_src") as c:
        c.attr(label="  Bronze · as delivered  ", labeljust="l",
               style="rounded", color=SKY, fontname=MONO, fontsize="11",
               fontcolor=RIDGE, bgcolor="white")
        c.attr("node", fillcolor="white", penwidth="1.2")
        for key, title, sub in [
            ("src_items", "transaction_items", "128.6M lines"),
            ("src_gtin", "master_gtin", "99k products"),
            ("src_shopper", "shopper_id", "71.8M payments"),
            ("src_stores", "stores_information", "36.9k stores"),
            ("src_disc", "discounts", "17.4M applications"),
        ]:
            node(c, key, title, sub)
        c.body.append(
            "{rank=same; src_items; src_gtin; src_shopper; src_stores; src_disc}")

    # ---- silver: cleaned and conformed ----------------------------
    with g.subgraph(name="cluster_silver") as c:
        c.attr(label="  Silver · cleaned  ", labeljust="l", style="rounded",
               color=TEAL, fontname=MONO, fontsize="11", fontcolor=TEAL,
               bgcolor="white")
        node(c, "line_clean", "line_clean", "one row per sellable line")
        node(c, "fuel_anchor", "fuel_anchor", "one row per fuel transaction")
        node(c, "store_tx", "store_tx", "one row per in-store transaction")
        c.body.append("{rank=same; fuel_anchor; store_tx}")

    # ---- gold: the unit of analysis -------------------------------
    node(g, "visit", "visit", "one row per fuel visit · <b>the label</b>",
         color=ORANGE, fillcolor="#FEF5E9", penwidth="2.2")

    # ---- features -------------------------------------------------
    with g.subgraph(name="cluster_feat") as c:
        c.attr(label="  Features  ", labeljust="l", style="rounded",
               color=SKY, fontname=MONO, fontsize="11", fontcolor=RIDGE,
               bgcolor="white")
        node(c, "feat_ctx", "feat_context", "visit · known at the pump")
        node(c, "feat_hist", "feat_history", "visit × category · point-in-time")
        node(c, "als_int", "als_interactions", "shopper × sku@daypart")
        c.body.append("{rank=same; feat_ctx; feat_hist; als_int}")

    # ---- training sets --------------------------------------------
    node(g, "train_relevance", "train_relevance", "36.1M rows · 6 labels")

    # ---- models ---------------------------------------------------
    with g.subgraph(name="cluster_models") as c:
        c.attr(label="  Fitted models  ", labeljust="l", style="rounded",
               color=TEAL, fontname=MONO, fontsize="11", fontcolor=TEAL,
               bgcolor="white")
        c.attr("node", color=TEAL, fillcolor="#EEF5F8", penwidth="1.8")
        node(c, "m_relevance", "category_relevance", "6 × calibrated GBT")
        node(c, "m_affinity", "sku_affinity", "ALS, implicit feedback")
        c.body.append("{rank=same; m_relevance; m_affinity}")

    # ---- serving --------------------------------------------------
    node(g, "offer_cat", "offer_catalog",
         "sku · margin · funding<br/><b>does not exist</b>", color=BAD,
         fillcolor="white", style="rounded,filled,dashed")
    node(g, "score_req", "score_request", "one row per live fill")
    # The complement table: a different question, built from the joint
    # baskets rather than from either model.
    node(g, "pair_affinity", "pair_affinity",
         "category × category · support,<br/>lift, cohort index, excess",
         color=TEAL, fillcolor="#EEF5F8", penwidth="1.8")
    # The batch twin of `decision`: every held-out fill, scored and written.
    node(g, "cust_scores", "customer_scores",
         "one row per customer-moment<br/><b>583,583 rows · 19 columns</b>",
         color=GOOD, fillcolor="#EDF7F2", penwidth="2")
    node(g, "decision", "decision",
         "candidate × fill · <b>stage one picks the<br/>category, "
         "stage two ranks inside it</b>",
         color=GOOD, fillcolor="#EDF7F2", penwidth="2.2")
    node(g, "validation", "validation",
         "calibration by decile ·<br/>hit@10 vs the best seller",
         color=GOOD, fillcolor="#EDF7F2", penwidth="1.8")
    node(g, "dec_log", "decision_log",
         "what we showed · <b>and with what probability</b>", color=ORANGE,
         fillcolor="#FEF5E9", penwidth="2")

    # ---- wiring ----------------------------------------------------
    g.edge("src_items", "line_clean")
    g.edge("src_gtin", "line_clean", label="GTIN")
    g.edge("line_clean", "fuel_anchor", label="category = FUEL")
    g.edge("line_clean", "store_tx", label="else")
    g.edge("src_shopper", "fuel_anchor", label="SHOPPER_ID")
    g.edge("src_shopper", "store_tx", style="dashed")
    g.edge("fuel_anchor", "visit")
    g.edge("store_tx", "visit", label="−15 / +45 min")

    g.edge("visit", "feat_ctx")
    g.edge("visit", "feat_hist")
    g.edge("src_stores", "feat_ctx", style="dashed")
    g.edge("store_tx", "als_int", style="dashed", constraint="false")

    g.edge("feat_ctx", "train_relevance")
    g.edge("feat_hist", "train_relevance")
    g.edge("train_relevance", "m_relevance")
    g.edge("als_int", "m_affinity")

    g.edge("m_relevance", "decision", label="P(category)")
    g.edge("m_affinity", "decision", label="top SKUs")
    g.edge("offer_cat", "decision", style="dashed", label="margin · funding")
    g.edge("score_req", "decision", style="dashed")
    g.edge("pair_affinity", "decision", style="dashed",
           label="the second item")
    g.edge("visit", "pair_affinity", style="dashed",
           label="training baskets")
    g.edge("m_relevance", "cust_scores", style="dashed")
    g.edge("m_affinity", "cust_scores", style="dashed")
    g.edge("pair_affinity", "cust_scores", style="dashed")
    g.edge("cust_scores", "validation")
    g.edge("visit", "validation", style="dashed", constraint="false",
           label="what actually\\nhappened", color=GOOD, fontcolor=GOOD)
    g.edge("decision", "dec_log")
    g.edge("dec_log", "visit", style="dashed", constraint="false",
           label="tomorrow's training data", color=ORANGE, fontcolor="#B5721F")

    return g


def write_partial(svg_path: Path) -> Path:
    """Wrap the SVG in a raw-HTML block Quarto can include."""
    svg = svg_path.read_text()
    svg = svg[svg.index("<svg"):]
    for attr in ("width", "height"):
        i = svg.index(f'{attr}="')
        j = svg.index('"', i + len(attr) + 2) + 1
        svg = svg[:i] + svg[j:]
    svg = svg.replace("<svg ", '<svg id="pipeline-svg" ', 1)
    out = svg_path.parent.parent / "_tables_diagram.qmd"
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
