"""Audit the five personas end to end: raw data -> prediction -> outcome.

Three questions per customer, in the order a sceptic would ask them:

    1. What does the raw data actually say about this person?
    2. What did we predict for them, out of customer_scores.parquet?
    3. What did they actually do in the held-out period?

Question 3 is the one that bites. The personas are chosen for being TYPICAL of
an archetype in the training period, with no requirement that they reappear
afterwards -- and most of them do not. Four of the five fuel fewer than twenty
times in eighteen months, so their validation-period record is one or two
fills or none at all. A single shopper cannot validate a probability.

So each tab carries both: the individual, which shows the mechanism, and the
whole COHORT of shoppers matching that archetype, which is what the predicted
rate can honestly be checked against.

Writes, for the pipeline3 page:
    slides/img/persona_<slug>.png   raw behaviour next to prediction vs actual
    slides/img/calibration.png      predicted vs actual over every scored row
    slides/_personas_audit.qmd      a tabset, one tab per customer

Run:  uv run --with polars --with scikit-learn --with joblib --with implicit \
          --with matplotlib python persona_audit.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import config as C
import personas as P
from scoring import slug
from serve import OfferEngine, daypart_of

SLIDES = Path(__file__).resolve().parent.parent.parent / "slides"
IMG = SLIDES / "img"

# Validated with the dataviz palette checker against the #F4F7F9 chart
# surface: lightness band, chroma floor, CVD separation, normal-vision floor
# and contrast all pass. The deck's navy fails the chroma floor (it reads as
# grey to the validator), so these are the nearest passing steps to it.
PRED = "#1878B8"
ACTUAL = "#D9731A"
INK = "#111111"
MUTED = "#5A6B78"
GRID = "#D8E0E6"
SURFACE = "#F4F7F9"

CATS = C.TARGET_CATEGORIES
SHORT = {
    "Packaged Beverages (Non-Alcoholic)": "Beverages",
    "Other Tobacco/Nicotine Products": "Other tobacco",
    "Cigarettes": "Cigarettes",
    "Candy": "Candy",
    "Salty Snacks": "Salty snacks",
    "Beer": "Beer",
}


def mask(sid: str) -> str:
    """Shopper ids are restricted. Publish a prefix, never the whole hash."""
    return f"{sid[:12]}…"


def fname(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


# ------------------------------------------------------------------ raw data
def history(vis: pl.DataFrame, sid: str, cut) -> pl.DataFrame:
    return (vis.filter((pl.col("shopper_id") == sid) & (pl.col("ts") < cut))
            .sort("ts"))


def future(vis: pl.DataFrame, sid: str, cut) -> pl.DataFrame:
    return (vis.filter((pl.col("shopper_id") == sid) & (pl.col("ts") >= cut))
            .sort("ts"))


def cat_rates(df: pl.DataFrame) -> dict[str, float]:
    if df.height == 0:
        return {c: 0.0 for c in CATS}
    return {c: float(df["cats"].list.contains(c).fill_null(False).mean())
            for c in CATS}


def skus_bought(sid: str, cut) -> pl.DataFrame:
    """The actual items this shopper bought before the cutoff."""
    return (pl.scan_parquet(C.ART / "line_clean.parquet")
            .filter((pl.col("shopper_id") == sid)
                    & (pl.col("ts") < cut)
                    & pl.col("category").is_in(CATS))
            .group_by("category", "brand")
            .agg(units=pl.len(), spend=pl.col("amount").sum().round(2))
            .sort("units", descending=True)
            .collect())


def als_view(ranker, sid: str, daypart: str, category: str,
             cut) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """What ALS believes about this shopper, next to what they actually bought.

    The raw dot product is not comparable between shoppers, and on the
    fallback path it is not a dot product at all -- it is a unit count. So
    everything here is also reported as a PERCENTILE within the category's
    candidate set for that daypart, which is comparable.
    """
    idx = np.asarray((ranker.buckets.get(daypart) or {}).get(category, []))
    if idx.size == 0:
        empty = pl.DataFrame()
        return empty, empty, 0
    raw = ranker._raw([sid], idx)[0]
    gtins = [ranker.items[i].rpartition("@")[0] for i in idx]
    order = np.argsort(-raw, kind="stable")
    rank = np.empty(len(raw), dtype=int)
    rank[order] = np.arange(1, len(raw) + 1)
    pct = 100.0 * (1.0 - (rank - 1) / len(raw))
    scored = pl.DataFrame({
        "gtin": gtins,
        "brand": [ranker.meta[g]["brand"] for g in gtins],
        "als": raw, "rank": rank, "pct": pct})

    mine = (pl.scan_parquet(C.ART / "line_clean.parquet")
            .filter((pl.col("shopper_id") == sid) & (pl.col("ts") < cut)
                    & (pl.col("category") == category))
            .group_by("gtin").agg(units=pl.len()).collect())
    bought = (scored.join(mine, on="gtin", how="inner")
              .sort("units", descending=True))
    top = (scored.sort("als", descending=True).head(5)
           .join(mine, on="gtin", how="left")
           .with_columns(units=pl.col("units").fill_null(0)))
    return bought, top, len(raw)


def als_figure(label: str, bought: pl.DataFrame, n_cand: int,
               category: str, personal: bool, path: Path) -> None:
    """Where their own SKUs sit in the ALS ordering.

    A dot plot on a log rank axis, not bars on a percentile axis: the
    percentiles all crowd into 96-99 and bars that near-identical say
    nothing, while a truncated bar axis would be a lie. Dots carry no
    zero-baseline obligation, so the log scale is free.
    """
    b = bought.head(10)
    fig, ax = plt.subplots(figsize=(7.4, max(2.2, 0.34 * b.height + 1.1)),
                           facecolor="white")
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    y = np.arange(b.height)
    ranks = b["rank"].to_list()
    ax.hlines(y, 1, ranks, color=GRID, lw=1.5, zorder=1)
    ax.scatter(ranks, y, s=90, color=PRED, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, [str(br or "?") for br in b["brand"].to_list()])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, n_cand * 1.15)
    ax.axvline(10, color=ACTUAL, lw=1.5, ls=(0, (4, 3)), zorder=2)
    # Below the last row, not above the first: at the top it collides with
    # the title.
    ax.text(10, b.height - 0.35, " top 10", color=ACTUAL, fontsize=8,
            ha="left", va="top")
    ax.set_xlabel(f"ALS rank among {n_cand:,} candidate SKUs in "
                  f"{SHORT.get(category, category)}  (1 = best, log scale)",
                  color=MUTED, fontsize=8)
    kind = "ALS factors" if personal else "popularity — no ALS history"
    ax.set_title(f"SKUs they actually bought, ranked by {kind}",
                 color=INK, fontsize=10, loc="left", weight="bold")
    for i, (r, u) in enumerate(zip(ranks, b["units"].to_list())):
        ax.annotate(f"{u}×", (r, i), textcoords="offset points",
                    xytext=(10, 0), va="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------- the cohort
SHORTNAME = {"never bought": "never in", "soda regular": "soda",
             "cigarette regular": "cigs", "salty snack regular": "salty",
             "beer regular": "beer"}


def pick_with_record(prof: pl.DataFrame, eng: OfferEngine, category: str | None,
                     taken: set[str], min_test: int = 8) -> str | None:
    """The same archetype, but restricted to shoppers who came back.

    personas.py picks whoever is most TYPICAL before the cutoff and never
    asks whether they reappear, which is why four of its five have no
    held-out record at all. This adds the one condition that makes a
    prediction checkable against an outcome for that individual.
    """
    in_als = pl.Series("s", list(eng.shopper_ix.keys())).implode()
    base = prof.filter((pl.col("visits") >= 8)
                       & (pl.col("test") >= min_test)
                       & ~pl.col("shopper_id").is_in(pl.Series("t", list(taken) or [""]).implode()))
    if category is None:
        cand = base.filter(pl.col("bought") == 0).sort(
            ["visits", "shopper_id"], descending=[True, False])
    else:
        others = [c for c in (P.BEV, P.CIG, P.SALT, P.BEER) if c != category]
        cand = (base.filter((pl.col(category) >= 0.4)
                            & pl.col("shopper_id").is_in(in_als))
                .with_columns(sep=pl.col(category) - pl.max_horizontal(others))
                .sort(["sep", "shopper_id"], descending=[True, False]))
    return None if cand.height == 0 else cand.row(0, named=True)["shopper_id"]


def cohort(vis: pl.DataFrame, cut, category: str | None,
           scores: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Every shopper matching this archetype, scored on the test period.

    The individual shows the mechanism; this shows whether the number is
    right. Same archetype definition personas.py uses to pick the one.
    """
    prior = vis.filter(pl.col("ts") < cut)
    agg = [pl.len().alias("visits"), pl.col("bought").sum().alias("inside")]
    agg += [pl.col("cats").list.contains(c).fill_null(False).mean().alias(c)
            for c in CATS]
    prof = prior.group_by("shopper_id").agg(agg)
    if category is None:
        who = prof.filter((pl.col("visits") >= 8) & (pl.col("inside") == 0))
    else:
        who = prof.filter((pl.col("visits") >= 8) & (pl.col(category) >= 0.4))
    rows = scores.join(who.select("shopper_id"), on="shopper_id", how="inner")
    return rows, who.height


def pred_vs_actual(rows: pl.DataFrame, vis: pl.DataFrame, cut) -> pl.DataFrame:
    """Mean predicted propensity against the observed rate, per category."""
    if rows.height == 0:
        return pl.DataFrame({"category": [], "predicted": [], "actual": []})
    # fuel_tx, not (shopper_id, ts): the pair is not unique.
    truth = vis.join(rows.select("fuel_tx"), on="fuel_tx", how="semi")
    out = []
    for c in CATS:
        p = float(rows[f"{slug(c)}_propensity"].mean())
        a = float(truth["cats"].list.contains(c).fill_null(False).mean()) \
            if truth.height else float("nan")
        out.append({"category": SHORT[c], "predicted": p, "actual": a})
    return pl.DataFrame(out)


# ------------------------------------------------------------------- plots
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def persona_figure(label: str, sid: str, prior: dict[str, float],
                   n_prior: int, own: pl.DataFrame, n_rows: int,
                   path: Path) -> None:
    """One customer's story on one axis: history -> prediction -> outcome.

    This replaced a two-panel figure whose left panel was *this shopper's*
    history in percent and whose right panel was the *cohort's* rates in
    percent. Two different populations on two axes that looked alike invites
    a comparison that is not valid, and half the figure was not about the
    customer whose tab it sat in.

    Encoding: prior is a hollow reference marker rather than a third hue --
    it is context, not a third competing series -- and the predicted/actual
    pair keeps the validated two-colour palette, joined by the gap so the
    error reads as a length.
    """
    fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor="white")
    style(ax)
    y = np.arange(len(CATS))
    names = [SHORT[c] for c in CATS]
    pri = np.array([100 * prior[c] for c in CATS])

    have = own.height > 0
    if have:
        idx = {r["category"]: r for r in own.iter_rows(named=True)}
        pred = np.array([100 * idx[SHORT[c]]["predicted"] for c in CATS])
        act = np.array([100 * idx[SHORT[c]]["actual"] for c in CATS])
        ax.hlines(y, np.minimum(pred, act), np.maximum(pred, act),
                  color=GRID, lw=3, zorder=1)
        ax.scatter(pred, y, s=150, color=PRED, zorder=3, label="we predicted",
                   edgecolor="white", linewidth=1.5)
        ax.scatter(act, y, s=70, color=ACTUAL, zorder=4, label="they did",
                   edgecolor="white", linewidth=1.2)
        top = max(pri.max(), pred.max(), act.max())
    else:
        top = max(pri.max(), 1.0)
    ax.scatter(pri, y, s=110, facecolor="none", edgecolor=MUTED, zorder=2,
               linewidth=1.6, label="before the cutoff")

    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(-max(2.0, top * 0.04), top * 1.2)
    ax.set_xlabel("% of fills", color=MUTED, fontsize=9)
    sub = (f"{n_rows} held-out" if have else "none held out")
    ax.set_title(f"{label} · {mask(sid)}   "
                 f"({n_prior:,} fills before the cutoff · {sub})",
                 color=INK, fontsize=10.5, loc="left", weight="bold", pad=26)
    leg = ax.legend(frameon=False, fontsize=8.5, ncols=3,
                    loc="lower left", bbox_to_anchor=(0, 1.005))
    for t in leg.get_texts():
        t.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def als_figure(label: str, bought: pl.DataFrame, n_cand: int,
               category: str, personal: bool, path: Path) -> None:
    """Where their own SKUs sit in the ALS ordering.

    A dot plot on a log rank axis, not bars on a percentile axis: the
    percentiles all crowd into 96-99 and bars that near-identical say
    nothing, while a truncated bar axis would be a lie. Dots carry no
    zero-baseline obligation, so the log scale is free.
    """
    b = bought.head(10)
    fig, ax = plt.subplots(figsize=(7.4, max(2.2, 0.34 * b.height + 1.1)),
                           facecolor="white")
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    y = np.arange(b.height)
    ranks = b["rank"].to_list()
    ax.hlines(y, 1, ranks, color=GRID, lw=1.5, zorder=1)
    ax.scatter(ranks, y, s=90, color=PRED, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, [str(br or "?") for br in b["brand"].to_list()])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.8, n_cand * 1.15)
    ax.axvline(10, color=ACTUAL, lw=1.5, ls=(0, (4, 3)), zorder=2)
    # Below the last row, not above the first: at the top it collides with
    # the title.
    ax.text(10, b.height - 0.35, " top 10", color=ACTUAL, fontsize=8,
            ha="left", va="top")
    ax.set_xlabel(f"ALS rank among {n_cand:,} candidate SKUs in "
                  f"{SHORT.get(category, category)}  (1 = best, log scale)",
                  color=MUTED, fontsize=8)
    kind = "ALS factors" if personal else "popularity — no ALS history"
    ax.set_title(f"SKUs they actually bought, ranked by {kind}",
                 color=INK, fontsize=10, loc="left", weight="bold")
    for i, (r, u) in enumerate(zip(ranks, b["units"].to_list())):
        ax.annotate(f"{u}×", (r, i), textcoords="offset points",
                    xytext=(10, 0), va="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------- the cohort
SHORTNAME = {"never bought": "never in", "soda regular": "soda",
             "cigarette regular": "cigs", "salty snack regular": "salty",
             "beer regular": "beer"}


def pick_with_record(prof: pl.DataFrame, eng: OfferEngine, category: str | None,
                     taken: set[str], min_test: int = 8) -> str | None:
    """The same archetype, but restricted to shoppers who came back.

    personas.py picks whoever is most TYPICAL before the cutoff and never
    asks whether they reappear, which is why four of its five have no
    held-out record at all. This adds the one condition that makes a
    prediction checkable against an outcome for that individual.
    """
    in_als = pl.Series("s", list(eng.shopper_ix.keys())).implode()
    base = prof.filter((pl.col("visits") >= 8)
                       & (pl.col("test") >= min_test)
                       & ~pl.col("shopper_id").is_in(pl.Series("t", list(taken) or [""]).implode()))
    if category is None:
        cand = base.filter(pl.col("bought") == 0).sort(
            ["visits", "shopper_id"], descending=[True, False])
    else:
        others = [c for c in (P.BEV, P.CIG, P.SALT, P.BEER) if c != category]
        cand = (base.filter((pl.col(category) >= 0.4)
                            & pl.col("shopper_id").is_in(in_als))
                .with_columns(sep=pl.col(category) - pl.max_horizontal(others))
                .sort(["sep", "shopper_id"], descending=[True, False]))
    return None if cand.height == 0 else cand.row(0, named=True)["shopper_id"]


def cohort(vis: pl.DataFrame, cut, category: str | None,
           scores: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Every shopper matching this archetype, scored on the test period.

    The individual shows the mechanism; this shows whether the number is
    right. Same archetype definition personas.py uses to pick the one.
    """
    prior = vis.filter(pl.col("ts") < cut)
    agg = [pl.len().alias("visits"), pl.col("bought").sum().alias("inside")]
    agg += [pl.col("cats").list.contains(c).fill_null(False).mean().alias(c)
            for c in CATS]
    prof = prior.group_by("shopper_id").agg(agg)
    if category is None:
        who = prof.filter((pl.col("visits") >= 8) & (pl.col("inside") == 0))
    else:
        who = prof.filter((pl.col("visits") >= 8) & (pl.col(category) >= 0.4))
    rows = scores.join(who.select("shopper_id"), on="shopper_id", how="inner")
    return rows, who.height


def pred_vs_actual(rows: pl.DataFrame, vis: pl.DataFrame, cut) -> pl.DataFrame:
    """Mean predicted propensity against the observed rate, per category."""
    if rows.height == 0:
        return pl.DataFrame({"category": [], "predicted": [], "actual": []})
    # fuel_tx, not (shopper_id, ts): the pair is not unique.
    truth = vis.join(rows.select("fuel_tx"), on="fuel_tx", how="semi")
    out = []
    for c in CATS:
        p = float(rows[f"{slug(c)}_propensity"].mean())
        a = float(truth["cats"].list.contains(c).fill_null(False).mean()) \
            if truth.height else float("nan")
        out.append({"category": SHORT[c], "predicted": p, "actual": a})
    return pl.DataFrame(out)


# ------------------------------------------------------------------- plots
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def overview_figure(rows: list[dict], path: Path) -> None:
    """Predicted against actual for every customer, on one chart.

    A dumbbell, not paired bars: the question is the GAP between two values
    on the same scale for each customer, and a dumbbell draws that gap as a
    length instead of making the eye compare two bar ends. One row per
    customer, sorted by actual rate so the ordering carries information.
    """
    rows = [r for r in rows if r["n"] > 0]
    fig, ax = plt.subplots(figsize=(8.0, 0.46 * len(rows) + 1.6),
                           facecolor="white")
    style(ax)
    y = np.arange(len(rows))
    pred = np.array([100 * r["pred"] for r in rows])
    act = np.array([100 * r["act"] for r in rows])
    ax.hlines(y, np.minimum(pred, act), np.maximum(pred, act),
              color=GRID, lw=3, zorder=1)
    # Different sizes on purpose. Where prediction and outcome coincide --
    # which is the best thing that can happen here -- equal markers would
    # hide one behind the other and the chart would look like missing data.
    ax.scatter(pred, y, s=150, color=PRED, zorder=3, label="predicted",
               edgecolor="white", linewidth=1.5)
    ax.scatter(act, y, s=70, color=ACTUAL, zorder=4, label="actual",
               edgecolor="white", linewidth=1.2)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-3, max(pred.max(), act.max()) * 1.18)
    ax.set_xlabel("% of their held-out fills that bought the category",
                  color=MUTED, fontsize=9)
    ax.set_title("Predicted vs actual, per customer",
                 color=INK, fontsize=11, loc="left", weight="bold", pad=26)
    # Above the plot, never inside it: at lower-right it sat on a data row.
    leg = ax.legend(frameon=False, fontsize=9, ncols=2,
                    loc="lower left", bbox_to_anchor=(0, 1.005))
    for t in leg.get_texts():
        t.set_color(MUTED)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['n']} fills", (max(pred[i], act[i]), i),
                    textcoords="offset points", xytext=(12, 0),
                    va="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def calibration_figure(scores: pl.DataFrame, vis: pl.DataFrame,
                       path: Path) -> pl.DataFrame:
    """Predicted vs actual by decile of predicted probability, all rows.

    This is the check the individual tabs cannot do: 583,583 scored fills
    against what those same fills actually contained.
    """
    cat = "Packaged Beverages (Non-Alcoholic)"
    col = f"{slug(cat)}_propensity"
    keys = scores.select("fuel_tx", col)
    truth = (vis.select("fuel_tx", "cats")
             .join(keys, on="fuel_tx", how="inner")
             .with_columns(hit=pl.col("cats").list.contains(cat)
                           .fill_null(False).cast(pl.Int8)))
    t = (truth.with_columns(
            bucket=(((pl.col(col).rank("ordinal") - 1) * 10)
                    // pl.len()).cast(pl.Int32))
         .group_by("bucket")
         .agg(predicted=pl.col(col).mean(), actual=pl.col("hit").mean(),
              n=pl.len())
         .sort("bucket"))

    fig, ax = plt.subplots(figsize=(5.6, 4.4), facecolor="white")
    style(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    lo, hi = 0, max(t["predicted"].max(), t["actual"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot(t["predicted"], t["actual"], color=PRED, lw=2, zorder=2,
            marker="o", ms=8, mfc=PRED, mec="white", mew=1.5)
    ax.set_xlabel("predicted P(buys a beverage)", color=MUTED, fontsize=9)
    ax.set_ylabel("observed rate", color=MUTED, fontsize=9)
    ax.set_title("Calibration on the held-out period",
                 color=INK, fontsize=11, loc="left", weight="bold")
    ax.text(0.97, 0.06, "dashed = perfect calibration", transform=ax.transAxes,
            ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return t


# -------------------------------------------------------------------- tables
def md_table(df: pl.DataFrame, headers: list[str] | None = None) -> str:
    if df.height == 0:
        return "_no rows_\n"
    cols = df.columns
    head = headers or cols
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, bool):
                cells.append("**yes**" if v else "no")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 1 else f"{v:,.2f}")
            elif isinstance(v, int):
                # Ids are labels, not quantities: no thousands separator.
                cells.append(f"{v:,}" if abs(v) < 1000 or v > 100_000
                             else str(v))
            else:
                cells.append("—" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def ranker_of(eng: OfferEngine):
    """The engine already built one; reuse it."""
    return eng.ranker


def main() -> None:
    eng = OfferEngine()
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    when = datetime.fromisoformat("2026-06-15T07:12")
    vis = pl.read_parquet(C.ART / "visit.parquet")
    scores = pl.read_parquet(C.ART / "customer_scores.parquet")
    print(f"[audit] {scores.height:,} scored rows in the held-out period")

    cal = calibration_figure(scores, vis, IMG / "calibration.png")

    # personas.py profiles the training period only; add the held-out count so
    # the second five can be chosen on having one.
    post = (vis.filter(pl.col("ts") >= cut).group_by("shopper_id")
            .agg(test=pl.len()))
    prof = (P.shopper_profiles(eng, cut).join(post, on="shopper_id", how="left")
            .with_columns(test=pl.col("test").fill_null(0)))

    # Two of each archetype: the one personas.py picks for being typical, and
    # one that also has a held-out record to be checked against.
    roster, taken = [], set()
    for label, category in P.PERSONAS:
        sid = P.pick(prof, eng, category)
        if sid:
            taken.add(sid)
            roster.append((label, category, sid, "typical"))
    for label, category in P.PERSONAS:
        sid = pick_with_record(prof, eng, category, taken)
        if sid:
            taken.add(sid)
            roster.append((label, category, sid, "with a record"))

    parts, summary, overview = ["::: {.panel-tabset}\n"], [], []
    for label, category, sid, kind in roster:
        prior, post_v = history(vis, sid, cut), future(vis, sid, cut)
        rates = cat_rates(prior)
        mine = scores.filter(pl.col("shopper_id") == sid).sort("ts")
        crows, nshop = cohort(vis, cut, category, scores)
        pva = pred_vs_actual(crows, vis, cut)
        own_pva = pred_vs_actual(mine, vis, cut) if mine.height else pl.DataFrame()
        png = IMG / f"persona_{fname(label)}_{fname(kind)}.png"
        persona_figure(label, sid, rates, prior.height, own_pva,
                       mine.height, png)
        print(f"[audit] {label:22}{kind:14} {mask(sid)} prior={prior.height:5} "
              f"test={post_v.height:4}")
        # The archetype's own category; the never-bought shopper has none,
        # so fall back to the category stage one offers them.
        focus = category or CATS[0]
        if mine.height:
            own = pred_vs_actual(mine, vis, cut)
            hit = own.filter(pl.col("category") == SHORT[focus])
            overview.append({
                "label": f"{SHORTNAME[label]} · {mask(sid)[:10]}",
                "pred": float(hit["predicted"][0]) if hit.height else 0.0,
                "act": float(hit["actual"][0]) if hit.height else 0.0,
                "n": mine.height})
        summary.append({"persona": label, "set": kind, "shopper": mask(sid),
                        "prior_fills": prior.height,
                        "held_out_fills": post_v.height,
                        "scored_rows": mine.height})

        parts.append(f"\n## {SHORTNAME[label]} · {post_v.height}\n")
        parts.append(f"\n**{label}** — `{mask(sid)}`. "
                     f"**{prior.height:,}** fills before the cutoff, "
                     f"**{post_v.height:,}** after it"
                     + (", so this one can be checked against its own outcome.\n"
                        if post_v.height else
                        " — typical of the archetype, but never seen again.\n"))
        parts.append(f"\n![](img/{png.name})\n")

        # --- 1. raw
        parts.append('\n::: {.callout-note collapse="true" title="1 · The raw '
                     'data we trained on"}\n')
        parts.append(f"\nEvery fuel visit before {C.TRAIN_CUTOFF}. This is all "
                     "the model was allowed to see.\n\n")
        shown = (prior.select("ts", "daypart", "fuel_grade", "store_id",
                              "fuel_spend", "bought", "cats")
                 .reverse()
                 .with_columns(
                     ts=pl.col("ts").dt.strftime("%Y-%m-%d %H:%M"),
                     store_id=pl.col("store_id").cast(pl.Utf8),
                     fuel_spend=pl.col("fuel_spend").round(2),
                     cats=pl.col("cats").cast(pl.List(pl.Utf8)).list.join(", ").fill_null("—")))
        parts.append('\n::: {.scroll-table}\n')
        parts.append(md_table(shown, ["when", "daypart", "grade", "store",
                                      "fuel $", "went in", "categories"]))
        parts.append(":::\n")
        parts.append(f"\n[All {prior.height:,} fills, newest first — scroll "
                     "the box. Their category rates are the left-hand chart "
                     'above.]{.muted style="font-size:0.8em"}\n')
        sk = skus_bought(sid, cut)
        if sk.height:
            parts.append(f"\nThe actual items behind those baskets — "
                         f"**{sk.height:,}** distinct SKUs:\n\n")
            parts.append('::: {.scroll-table}\n')
            parts.append(md_table(sk, ["category", "brand", "units",
                                       "spend $"]))
            parts.append(":::\n")
        parts.append("\n:::\n")

        # --- 2. prediction
        parts.append('\n::: {.callout-note collapse="true" title="2 · What we '
                     'predicted"}\n')
        d = eng.decide(sid, when, "REGULAR")
        props = eng.category_probs(sid, when, "REGULAR")
        parts.append("\nAt one fixed moment — **2026-06-15 07:12, REGULAR** — "
                     "so the only thing that varies across these tabs is who "
                     "they are:\n\n")
        parts.append(md_table(pl.DataFrame(
            [{"category": SHORT[c], "P(buys on this fill)": props[c]}
             for c in CATS]).sort("P(buys on this fill)", descending=True)))
        best = d.show
        parts.append(f"\n**Stage one picks {d.category}.** "
                     + (f"Stage two offers **{best.brand}** "
                        f"(EV ${best.expected_value:.5f}).\n"
                        if best else "Stage two shows **nothing** — every "
                        "candidate prices below the threshold.\n"))
        if mine.height:
            pcols = [f"{slug(c)}_propensity" for c in CATS]
            parts.append("\nAnd the rows actually written to "
                         "`customer_scores.parquet` for their held-out "
                         "fills:\n\n")
            parts.append('::: {.scroll-table}\n')
            parts.append(md_table(
                mine.select("ts", "daypart", "prob_entering", "offer_category",
                            *pcols)
                .with_columns(ts=pl.col("ts").dt.strftime("%m-%d %H:%M")),
                ["when", "daypart", "P(enters)", "offer category"]
                + [SHORT[c] for c in CATS]))
            parts.append(":::\n")
            top = mine["top_skus"][0]
            if top is not None and len(top):
                parts.append("\nRanked SKUs inside that category:\n\n")
                parts.append(md_table(
                    top.struct.unnest().select(
                        "rank", "brand", pl.col("affinity").round(2)
                        .cast(pl.Utf8), "p_sku", "p_category", "p_buys")
                    .head(3)))
            s2, sc2 = mine["second_category_skus"][0], mine["second_category"][0]
            if s2 is not None and len(s2):
                b = s2.struct.unnest()
                parts.append(f"\nSecond item: **{b['brand'][0]}** ({sc2}), "
                             f"cohort index {b['cohort_index'][0]:.2f}, "
                             f"headroom {b['headroom'][0]:.0%}\n")
            else:
                parts.append(f"\nSecond item: **none** "
                             f"({sc2 or 'nothing clears the gate'})\n")
        else:
            parts.append("\n_No scored rows: this shopper has no held-out "
                         "fills._\n")
        parts.append("\n:::\n")

        # --- 4. ALS against their own record
        # Which category to look at: the one they actually buy. For the
        # never-bought shopper there is none, so fall back to what stage one
        # would offer them.
        own_cat = max(CATS, key=lambda c: rates[c])
        if rates[own_cat] == 0:
            own_cat = d.category or CATS[0]
        bought_df, top_df, ncand = als_view(ranker_of(eng), sid, "morning",
                                            own_cat, cut)
        personal = sid in eng.shopper_ix
        parts.append('\n::: {.callout-note collapse="true" title="4 · What ALS '
                     'believes, against what they bought"}\n')
        if ncand == 0:
            parts.append(f"\n_No morning candidates in {own_cat}._\n")
        else:
            apng = IMG / f"als_{fname(label)}_{fname(kind)}.png"
            if bought_df.height:
                als_figure(label, bought_df, ncand, own_cat, personal, apng)
            elif apng.exists():
                apng.unlink()          # nothing to plot; don't ship a blank
            parts.append(f"\nScored over the **{ncand:,}** SKUs ALS carries "
                         f"for *{SHORT.get(own_cat, own_cat)}* in the morning"
                         + ("." if personal else
                            ". This shopper is **not** in the factor matrix, "
                            "so these are popularity counts, not ALS scores "
                            "— the fallback path, and the honest answer for "
                            "someone with no history.") + "\n")
            if bought_df.height:
                parts.append(f"\n![](img/{apng.name})\n")
                parts.append("\nThe SKUs they bought, and where ALS puts "
                             "them:\n\n")
                parts.append('::: {.scroll-table}\n')
                parts.append(md_table(
                    bought_df.select(
                        "brand", "units",
                        pl.col("als").round(3).cast(pl.Utf8),
                        "rank", pl.col("pct").round(1)),
                    ["brand", "they bought", "ALS score", "rank", "percentile"]))
                parts.append(":::\n")
                hits = int((bought_df["rank"] <= 10).sum())
                parts.append(f"\n**{hits} of {bought_df.height}** of their own "
                             f"SKUs land in the top 10 of {ncand:,}.\n")
            else:
                parts.append("\nThey never bought anything in this category, "
                             "so there is nothing to line ALS up against.\n")
            parts.append("\nAnd what ALS ranks highest for them:\n\n")
            parts.append(md_table(
                top_df.select("brand", pl.col("als").round(3).cast(pl.Utf8),
                              "units"),
                ["brand", "ALS score", "times they bought it"]))
            parts.append("\n[The raw score is a dot product on the ALS path "
                         "and a unit count on the fallback path, so it is not "
                         "one scale — the percentile is the comparable "
                         'column.]{.muted style="font-size:0.8em"}\n')
        parts.append("\n:::\n")

        # --- 3. outcome
        parts.append('\n::: {.callout-note collapse="true" title="3 · What '
                     'actually happened"}\n')
        if post_v.height == 0:
            parts.append("\nThis shopper **never fuelled again** after the "
                         "cutoff, so there is nothing to score them against. "
                         "That is the ordinary case for a customer who fuels "
                         f"{prior.height} times in eighteen months, and it is "
                         "why the cohort below carries the check.\n")
        else:
            act = (post_v.select("ts", "daypart", "bought", "cats")
                   .with_columns(ts=pl.col("ts").dt.strftime("%m-%d %H:%M"),
                                 cats=pl.col("cats").cast(pl.List(pl.Utf8)).list.join(", ")
                                 .fill_null("—")))
            parts.append(f"\nTheir own held-out fills — "
                         f"**{post_v['bought'].sum()} of {post_v.height}** "
                         "ended inside:\n\n")
            parts.append('::: {.scroll-table}\n')
            parts.append(md_table(act, ["when", "daypart", "went in",
                                        "what they bought"]))
            parts.append(":::\n")
        if mine.height:
            own = pred_vs_actual(mine, vis, cut)
            parts.append(f"\nTheir own predicted rate against their own "
                         f"outcome, over those {mine.height} fills. "
                         "At this sample size the gaps are noise as much as "
                         "signal — which is the point of the cohort below:\n\n")
            parts.append(md_table(
                own.with_columns(gap=(pl.col("actual") - pl.col("predicted"))),
                ["category", "mean predicted", "actual", "gap"]))
        if pva.height:
            parts.append(f"\nAnd across **{nshop:,} shoppers** matching this "
                         f"archetype, over **{crows.height:,}** held-out "
                         "fills — enough rows for the rate to mean "
                         "something:\n\n")
            parts.append(md_table(
                pva.with_columns(gap=(pl.col("actual") - pl.col("predicted"))),
                ["category", "mean predicted", "actual", "gap"]))
        parts.append("\n:::\n")
    parts.append("\n:::\n")

    overview.sort(key=lambda r: -r["act"])
    overview_figure(overview, IMG / "persona_overview.png")
    print(f"[audit] wrote {IMG / 'persona_overview.png'}")
    (SLIDES / "_personas_audit.qmd").write_text("".join(parts))
    print(f"[audit] wrote {SLIDES / '_personas_audit.qmd'}")
    with pl.Config(tbl_width_chars=150, tbl_rows=12):
        print(pl.DataFrame(summary))
    with pl.Config(tbl_width_chars=120):
        print(cal.with_columns(gap=(pl.col("actual") - pl.col("predicted"))))


if __name__ == "__main__":
    main()
