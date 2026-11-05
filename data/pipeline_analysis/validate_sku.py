"""Does SKU Affinity actually pick the item they bought?

`validate.py` checks Category Relevance by calibration: predicted P(category)
against the observed rate. That says nothing about stage two. This answers the
stage-two question directly, on the held-out period:

    when a customer bought something in the category we offered,
    was the item they bought in our top k?

The comparison that matters is **on the same rows**. Comparing ALS rows
against fallback rows compares two different populations -- the ALS path only
exists for shoppers with history, who are heavier buyers to begin with, and
would flatter ALS for reasons that have nothing to do with the model. So for
every ALS-eligible fill this scores both rankings over the identical candidate
set and the identical basket.

The `hit@10 16.2% vs 5.3%` in config.py came from a run that no longer exists.
This script is where that claim is supposed to come from.

Run:  uv run --with polars --with scikit-learn --with implicit --with joblib \
          python validate_sku.py
"""

from __future__ import annotations

import joblib
import numpy as np
import polars as pl

import config as C
from scoring import SkuRanker

KS = (1, 3, 5, 10)


def basket_gtins() -> pl.DataFrame:
    """fuel_tx -> the GTINs actually bought in that visit's window.

    Rebuilt with the same bucketed window join build_tables.py uses, because
    visit.parquet keeps the categories but drops the items.
    """
    lines = pl.read_parquet(
        C.ART / "line_clean.parquet",
        columns=["tx", "shopper_id", "ts", "category", "gtin"])
    fuel = (lines.filter(pl.col("category") == "FUEL")
            .group_by("tx").agg(shopper_id=pl.col("shopper_id").first(),
                                ts=pl.col("ts").min())
            .rename({"tx": "fuel_tx"}))
    store = (lines.filter(~pl.col("category").is_in(C.NOT_A_PURCHASE)
                          & pl.col("gtin").is_not_null())
             .group_by("tx").agg(shopper_id=pl.col("shopper_id").first(),
                                 ts=pl.col("ts").min(),
                                 gtins=pl.col("gtin"),
                                 cats=pl.col("category")))
    fb = fuel.with_columns(bucket=pl.col("ts").dt.truncate("1h"))
    sb = store.with_columns(bucket=pl.col("ts").dt.truncate("1h"))
    shifted = pl.concat([sb.with_columns(bucket=pl.col("bucket")
                                         + pl.duration(hours=k))
                         for k in (-1, 0, 1)])
    delta = (pl.col("ts_store") - pl.col("ts")).dt.total_minutes()
    return (fb.join(shifted.select("shopper_id", "bucket", "gtins", "cats",
                                   ts_store="ts"),
                    on=["shopper_id", "bucket"], how="inner")
            .filter((delta >= -C.WINDOW_BEFORE) & (delta <= C.WINDOW_AFTER))
            .group_by("fuel_tx")
            .agg(gtins=pl.col("gtins")
                 .list.explode(keep_nulls=False, empty_as_null=False).unique(),
                 bought_cats=pl.col("cats")
                 .list.explode(keep_nulls=False, empty_as_null=False).unique()))


def popularity_topk(ranker: SkuRanker, daypart: str, category: str,
                    k: int) -> list[str]:
    """What the best-seller fallback would show for this daypart+category."""
    idx = ranker.buckets.get(daypart, {}).get(category)
    if idx is None or len(idx) == 0:
        return []
    pop = ranker.popularity[idx]
    order = np.argsort(-pop, kind="stable")[:k]
    return [ranker.items[idx[i]].split("@")[0] for i in order]


def main() -> None:
    print("[sku] loading")
    aff = joblib.load(C.ART / "model_sku_affinity.joblib")
    catalog = pl.read_csv(C.ART / "offer_catalog.csv",
                          schema_overrides={"gtin": pl.Utf8})
    ranker = SkuRanker(aff, catalog)
    scores = pl.read_parquet(
        C.ART / "customer_scores.parquet",
        columns=["fuel_tx", "daypart", "offer_category", "personalised",
                 "top_skus"])
    baskets = basket_gtins()
    print(f"[sku] {scores.height:,} scored fills, "
          f"{baskets.height:,} with a basket")

    df = scores.join(baskets, on="fuel_tx", how="inner")
    # Only fills where they bought something IN the category we offered: if
    # they bought nothing there, stage two had nothing to get right and
    # scoring it would just measure stage one again.
    df = df.with_columns(
        in_cat=pl.col("bought_cats").list.contains(pl.col("offer_category")))
    df = df.filter(pl.col("in_cat"))
    print(f"[sku] {df.height:,} fills bought inside the offered category "
          f"-- the rows stage two can be judged on")
    if df.height == 0:
        raise SystemExit("no evaluable rows")

    # Our ranking, and what popularity alone would have said on the same row.
    pop_cache: dict[tuple[str, str], list[str]] = {}
    rows = df.select("daypart", "offer_category", "personalised",
                     "top_skus", "gtins").to_dicts()
    kmax = max(KS)
    ours = {k: [] for k in KS}
    pops = {k: [] for k in KS}
    pers = []
    for r in rows:
        truth = set(r["gtins"])
        ranked = [c["gtin"] for c in r["top_skus"]]
        key = (r["daypart"], r["offer_category"])
        if key not in pop_cache:
            pop_cache[key] = popularity_topk(ranker, *key, kmax)
        pbase = pop_cache[key]
        pers.append(bool(r["personalised"]))
        for k in KS:
            ours[k].append(bool(truth & set(ranked[:k])))
            pops[k].append(bool(truth & set(pbase[:k])))

    pers = np.array(pers)
    print(f"[sku] personalised (ALS) path on {pers.mean():.1%} of them\n")

    def block(title, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"{title}: no rows\n")
            return
        print(f"{title}  ({n:,} fills)")
        print(f"  {'k':>3}  {'SKU Affinity':>13}  {'best seller':>12}  {'lift':>6}")
        for k in KS:
            a = np.array(ours[k])[mask].mean()
            b = np.array(pops[k])[mask].mean()
            lift = (a / b) if b > 0 else float("nan")
            print(f"  {k:>3}  {a:>12.1%}  {b:>11.1%}  {lift:>5.2f}x")
        print()

    block("ALL evaluable fills", np.ones(len(pers), dtype=bool))
    block("Personalised rows only (ALS actually ran)", pers)
    block("Fallback rows (popularity was the ranking)", ~pers)

    # Per category, on the personalised rows: where does ALS earn its place?
    print("Personalised rows, hit@10 by offered category")
    cat = np.array([r["offer_category"] for r in rows])
    h10, p10 = np.array(ours[10]), np.array(pops[10])
    out = []
    for c in sorted(set(cat)):
        m = (cat == c) & pers
        if m.sum() < 200:
            continue
        out.append({"category": c[:34], "fills": int(m.sum()),
                    "als": float(h10[m].mean()),
                    "best_seller": float(p10[m].mean()),
                    "lift": float(h10[m].mean() / p10[m].mean())
                    if p10[m].mean() else float("nan")})
    with pl.Config(tbl_width_chars=120):
        print(pl.DataFrame(out).sort("fills", descending=True))





# --------------------------------------------------------------- the pair
def popular_second(baskets: pl.DataFrame, anchor: str) -> str | None:
    """The second category a popularity ranking would have chosen.

    This is the baseline the pairing argument is actually against: the claim
    is that ranking on excess beats ranking on how often the category shows
    up next to the anchor. Candy is the case in point -- 18% of every second
    item in a drink basket, and an index of 1.07.
    """
    sub = baskets.filter(pl.col("cats").list.contains(anchor))
    if sub.height == 0:
        return None
    counts = (sub.select("cats").explode("cats")
              .filter((pl.col("cats") != anchor)
                      & ~pl.col("cats").is_in(C.NOT_OFFERABLE
                                              + C.NOT_PAIRABLE))
              .group_by("cats").len().sort("len", descending=True))
    return None if counts.height == 0 else counts["cats"][0]


def validate_pair() -> None:
    """Did the SECOND item land, on fills where the anchor did?

    The anchor is checked by hit@k; until now the pair offer was not checked
    at all, which left the boldest recommendation in the write-ups as the
    least tested. Scored only on fills where they actually bought the anchor
    category -- that is the population a second item is offered to.
    """
    print("\n[pair] second-item offer, on the held-out period")
    scores = pl.read_parquet(
        C.ART / "customer_scores.parquet",
        columns=["fuel_tx", "offer_category", "second_category"])
    baskets = basket_gtins().rename({"bought_cats": "cats"})
    train = pl.read_parquet(C.ART / "visit.parquet",
                            columns=["ts", "cats", "bought"])
    train = train.filter(pl.col("bought")
                         & (pl.col("ts") < pl.lit(C.TRAIN_CUTOFF)
                            .str.to_datetime())).select("cats")

    df = (scores.join(baskets, on="fuel_tx", how="inner")
          .filter(pl.col("cats").list.contains(pl.col("offer_category"))
                  & pl.col("second_category").is_not_null()))
    print(f"[pair] {df.height:,} fills bought the anchor AND carried a "
          f"second-item offer")
    if df.height == 0:
        return

    pop = {a: popular_second(train, a)
           for a in df["offer_category"].unique().to_list()}
    rows = df.select("offer_category", "second_category", "cats").to_dicts()
    ours = np.array([r["second_category"] in set(r["cats"]) for r in rows])
    base = np.array([(pop.get(r["offer_category"]) or "") in set(r["cats"])
                     for r in rows])
    print(f"  our pair offer landed   {ours.mean():>7.1%}")
    print(f"  popularity baseline     {base.mean():>7.1%}"
          f"   ({ours.mean() / base.mean():.2f}x)" if base.mean() else "")
    anchors = np.array([r["offer_category"] for r in rows])
    out = []
    for a in sorted(set(anchors)):
        m = anchors == a
        if m.sum() < 200:
            continue
        out.append({"anchor": a[:30], "fills": int(m.sum()),
                    "our_pair": pop.get(a) and ours[m].mean(),
                    "popularity_pick": pop.get(a),
                    "baseline": base[m].mean()})
    with pl.Config(tbl_width_chars=130):
        print(pl.DataFrame(out))


def _run_all() -> None:
    main()
    validate_pair()


if __name__ == "__main__":
    _run_all()
