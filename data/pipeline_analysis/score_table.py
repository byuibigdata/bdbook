"""Score customers in bulk and write one wide row per customer-moment.

    fuel_tx | shopper_id | store_id | ts | daypart | fuel_grade | prior_attach
    prob_entering
    <category>_propensity  x6
    personalised | offer_category | second_category
    top_skus : list[struct{rank, gtin, brand, category,
                           affinity, p_sku, p_category, p_buys}]
    second_category_skus : list[struct{rank, gtin, brand, category,
                                       affinity, p_sku,
                                       cohort_index, excess, headroom}]

    affinity    raw ALS dot product, or a unit count on the fallback path --
                read it with `personalised`, it is not one scale
    p_sku       P(this SKU | this category), normalised within the category
    p_category  P(buys this category on this fill), from Category Relevance
    p_buys      p_category x p_sku -- P(buys this exact SKU on this fill)

`offer_category` is stage one of the decision: the most likely category we
are allowed to show. Every SKU in `top_skus` comes from that category and
no other, which is why the column is worth carrying even though it is
recoverable from the propensities.

`second_category_skus` is the PAIR offer, and it answers a different
question. `top_skus` is what they were going to buy anyway, which makes it
the worst thing to discount. The second category is the complement a
discount might actually add to the basket: picked from the cohort table by
excess x headroom, then its own SKUs ranked by ALS affinity.

It carries no `p_category` and no `p_buys`. The pair category has no
Category Relevance model -- it is usually outside the six -- so there is no
P(category) for it, and a product built from a placeholder would look like a
measurement. `cohort_index` and `excess` are its evidence instead:

    cohort_index  this cohort's per-basket rate of the pair, over the rate
                  among shoppers who buy JUST AS OFTEN but not the anchor
    excess        the same comparison as a difference, in attach points
    headroom      1 - this shopper's own prior rate of that category

The list is EMPTY when nothing clears the complement gate, which is a real
answer: for an alcohol anchor there is no second item worth showing. See
`config.NOT_PAIRABLE` and the pairing section of the README.

`prob_entering` is DERIVED, not modelled. The attach model was dropped
because it added no accuracy, so this is the independence product of the six
category probabilities:

    P(enters) = 1 - prod(1 - p_category)

Against a dedicated attach model that scores AUC 0.827, this reaches 0.822
and sits within a point on the mean. Use it for reporting; do NOT multiply it
back into the expected value, or the category term is counted twice.

The ranking itself is not implemented here -- it lives in `scoring.py`, which
`serve.py` also calls. Two copies of it drifted twice, and both times the
batch table disagreed with the live answer.

Run:  uv run --with polars --with scikit-learn --with implicit \
          python score_table.py --limit 20000     # --limit 0 for everything
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import polars as pl

import config as C
import scoring
from pairing import PAIR_TABLE, SECOND_SKU_FIELDS, BatchPairer, own_shares
from scoring import SkuRanker, slug

# Probabilities only. The catalog's margins are average price times an
# assumed 40% -- the PDI data has no cost column -- so every dollar figure
# downstream of them is an assumption dressed as a measurement. They stay in
# `negotiate.py`, which says so out loud.
TOP_SKU_FIELDS = ["rank", "gtin", "brand", "category",
                  "affinity", "p_sku", "p_category", "p_buys"]


def log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    print(f"[score] {msg}" + (f"   ({now - t0:.1f}s)" if t0 else ""), flush=True)
    return now


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=20_000,
                    help="customer-moments to score; 0 means the whole period")
    ap.add_argument("--top", type=int, default=10, help="SKUs per row")
    ap.add_argument("--out", default="customer_scores.parquet")
    args = ap.parse_args()

    t = log("loading models and features")
    rel = joblib.load(C.ART / "model_category_relevance.joblib")
    aff = joblib.load(C.ART / "model_sku_affinity.joblib")
    # The FULL catalog goes to the ranker: it decides for itself what is
    # showable, and normalises P(sku | category) over every SKU in the
    # category rather than only the ones we happen to be allowed to show.
    catalog = pl.read_csv(C.ART / "offer_catalog.csv",
                          schema_overrides={"gtin": pl.Utf8})

    visit = pl.read_parquet(C.ART / "visit.parquet")
    hist = pl.read_parquet(C.ART / "feat_history.parquet")
    wide = hist.pivot(on="category", index="fuel_tx",
                      values=["prior_n", "bought_before", "prior_share"],
                      aggregate_function="first")
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    # Score the held-out period: real customers at real moments the models
    # have never seen.
    df = visit.filter(pl.col("ts") >= cut).join(wide, on="fuel_tx", how="left")
    if args.limit and df.height > args.limit:
        df = df.sample(args.limit, seed=C.RANDOM_SEED)
    t = log(f"{df.height:,} customer-moments from the test period", t)

    X = scoring.design_matrix(df, rel["cat_maps"], rel["cols"])
    props = scoring.category_probs(rel["models"], X)
    entering = scoring.prob_entering(props)
    t = log("category propensities scored", t)

    ranker = SkuRanker(aff, catalog)
    dayparts = df["daypart"].to_list()
    offer_category = ranker.choose_category(dayparts, props)
    picks = ranker.rank(df["shopper_id"].to_list(), dayparts, props,
                        top=args.top)
    # Trim the candidate dicts to the published struct. `discount` and
    # `our_cost` stay out: both are recoverable from funded_share, and the
    # table is wide enough already.
    picks = [[{k: (round(c[k], 6) if isinstance(c[k], float) else c[k])
               for k in TOP_SKU_FIELDS} for c in row] for row in picks]
    t = log(f"top {args.top} SKUs per row", t)

    # ---- the second item ------------------------------------------------
    # A different question from top_skus, so a different table feeds it.
    second_cat: list[str | None] = [None] * df.height
    second: list[list[dict]] = [[] for _ in range(df.height)]
    pair_path = C.ART / PAIR_TABLE
    if not pair_path.exists():
        log(f"no {PAIR_TABLE} -- second_category_skus will be empty. "
            f"run: python pairing.py --build")
    else:
        pairer = BatchPairer(ranker, pl.read_parquet(pair_path),
                             set(catalog.filter(pl.col("in_stock")
                                                & pl.col("offerable"))
                                 ["category"].to_list()))
        cand_cats = pairer.candidate_cats(offer_category)
        # Point-in-time, over the WHOLE visit table rather than the scored
        # slice: a shopper's history lives before the test period.
        # Left-joined FROM df so the rows stay in df order -- choose() and
        # skus() index into this by position.
        own = (df.select("fuel_tx")
               .join(own_shares(visit, cand_cats), on="fuel_tx", how="left")
               .fill_null(0.0))
        second_cat, p_idx, p_exc, p_hr = pairer.choose(offer_category, own)
        second = pairer.skus(df["shopper_id"].to_list(), dayparts,
                             second_cat, p_idx, p_exc, p_hr, top=args.top)
        n_with = sum(1 for r in second if r)
        t = log(f"second item: {n_with:,} of {df.height:,} rows have a pair "
                f"({len(cand_cats)} candidate categories)", t)

    out = pl.DataFrame({
        # The primary key. Without it the only way back to the visit table is
        # (shopper_id, ts), which is not unique -- a shopper can have two fuel
        # transactions on the same timestamp, and that silently fans out any
        # join an auditor writes.
        "fuel_tx": df["fuel_tx"],
        "shopper_id": df["shopper_id"],
        "store_id": df["store_id"],
        "ts": df["ts"],
        "daypart": df["daypart"],
        "fuel_grade": df["fuel_grade"],
        "prior_attach": df["prior_attach"],
        # Clip AFTER rounding. A category saturated at 1 - 1e-5 rounds to a
        # flat 1.0 at five decimals, and a reported certainty of 1.0 is a
        # claim the model is not entitled to make.
        "prob_entering": np.clip(np.round(entering, 5),
                                 scoring.P_FLOOR, 1 - scoring.P_FLOOR),
        "offer_category": offer_category,
        "personalised": ranker.personalised(df["shopper_id"].to_list()),
        **{f"{slug(cat)}_propensity": np.clip(np.round(p, 5),
                                              scoring.P_FLOOR,
                                              1 - scoring.P_FLOOR)
           for cat, p in props.items()},
        "top_skus": picks,
        "second_category": second_cat,
        "second_category_skus": second,
    }).sort("prob_entering", descending=True)

    path = C.ART / args.out
    out.write_parquet(path)
    log(f"wrote {path}  ({out.height:,} rows, {len(out.columns)} columns)")

    with pl.Config(tbl_width_chars=150, fmt_str_lengths=22, tbl_rows=6):
        print("\n" + str(out.select(
            "shopper_id", "store_id", "daypart", "prob_entering",
            "offer_category", "personalised",
            "packaged_beverages_non_alcoholic_propensity",
            "salty_snacks_propensity", "beer_propensity").head(6)))
    sec = out.filter(pl.col("second_category_skus").list.len() > 0)
    if sec.height:
        print(f"\nsecond_category_skus for {sec['shopper_id'][0][:12]}… "
              f"(anchor {sec['offer_category'][0]} -> "
              f"pair {sec['second_category'][0]}):")
        with pl.Config(tbl_rows=5, tbl_width_chars=145, fmt_str_lengths=20):
            print(sec["second_category_skus"][0].struct.unnest()
                  .select(SECOND_SKU_FIELDS).head(5))
    print(f"\ntop_skus for {out['shopper_id'][0][:12]}… "
          f"(prob_entering {out['prob_entering'][0]:.4f}, "
          f"category {out['offer_category'][0]}):")
    with pl.Config(tbl_rows=10, tbl_width_chars=135, fmt_str_lengths=24):
        print(out["top_skus"][0].struct.unnest().select(TOP_SKU_FIELDS))


if __name__ == "__main__":
    main()
