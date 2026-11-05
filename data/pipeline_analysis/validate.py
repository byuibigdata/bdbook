"""Check the pipeline before anyone trusts a number from it.

Every check fails loudly with the actual value, because a pipeline that
silently degrades is worse than one that stops. Exits non-zero if anything
fails, so it can gate a scheduled run.

Run:  uv run --with polars --with scikit-learn --with joblib python validate.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import joblib
import numpy as np
import polars as pl

import config as C

PASS, FAIL, WARN = [], [], []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    bucket = PASS if ok else (WARN if warn_only else FAIL)
    bucket.append(f"{name}" + (f" — {detail}" if detail else ""))
    mark = "ok  " if ok else ("warn" if warn_only else "FAIL")
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- tables
def validate_tables() -> pl.DataFrame:
    print("\nTABLES")
    needed = ("visit.parquet", "feat_history.parquet",
              "als_interactions.parquet", "offer_catalog.csv")
    for f in needed:
        check(f"{f} exists", (C.ART / f).exists())
    if not all((C.ART / f).exists() for f in needed):
        sys.exit("\nbuild_tables.py has not been run")

    v = pl.read_parquet(C.ART / "visit.parquet")

    expected = {"fuel_tx", "shopper_id", "store_id", "ts", "fuel_spend",
                "fuel_grade", "cats", "bought", "store_spend", "hour",
                "weekday", "month", "daypart"}
    missing = expected - set(v.columns)
    check("visit schema", not missing, f"missing {missing}" if missing else "")

    check("fuel_tx unique", v["fuel_tx"].n_unique() == v.height,
          f"{v.height:,} rows, {v['fuel_tx'].n_unique():,} keys")

    for col in ("shopper_id", "store_id", "ts", "bought"):
        n = v[col].null_count()
        check(f"{col} has no nulls", n == 0, f"{n:,} nulls" if n else "")

    rate = v["bought"].mean()
    check("attach rate in a believable range", 0.05 < rate < 0.25,
          f"{rate:.1%} (PDI-wide run gave 12.8%)")

    # The label is only meaningful if SERVICES never reached it.
    leaked = (v.filter(pl.col("bought"))["cats"]
              .list.contains("SERVICES").fill_null(False).sum())
    check("SERVICES excluded from the label", leaked == 0,
          f"{leaked:,} visits label-leaked by card fees" if leaked else "")
    fuel_in = (v.filter(pl.col("bought"))["cats"]
               .list.contains("FUEL").fill_null(False).sum())
    check("FUEL excluded from the label", fuel_in == 0,
          f"{fuel_in:,} visits count fuel as their own outcome" if fuel_in else "")

    check("bought implies a basket",
          v.filter(pl.col("bought") & (pl.col("store_spend") <= 0)).height == 0)
    check("not-bought implies no spend",
          v.filter(~pl.col("bought") & (pl.col("store_spend") > 0)).height == 0)

    dp = set(v["daypart"].unique().to_list())
    check("dayparts are the three we chose", dp <= {"morning", "midday", "evening"},
          f"found {sorted(dp)}")
    return v


# -------------------------------------------------------------- leakage
def validate_leakage(v: pl.DataFrame) -> None:
    print("\nLEAKAGE")
    hist = pl.read_parquet(C.ART / "feat_history.parquet")

    check("feat_history covers every visit x category",
          hist.height == v.height * len(C.TARGET_CATEGORIES),
          f"{hist.height:,} vs expected {v.height * len(C.TARGET_CATEGORIES):,}")

    # A shopper's FIRST visit cannot have prior history. This is the single
    # most valuable check in the file: if it fails, every model is reading
    # its own answer.
    first = (v.sort("ts", "fuel_tx").group_by("shopper_id").first()
             .select("fuel_tx"))
    firsts = hist.join(first, on="fuel_tx", how="inner")
    bad_n = firsts.filter(pl.col("prior_n") != 0).height
    bad_h = firsts.filter(pl.col("bought_before") != 0).height
    check("first visit has prior_n = 0", bad_n == 0, f"{bad_n:,} rows" if bad_n else "")
    check("first visit has bought_before = 0", bad_h == 0,
          f"{bad_h:,} rows" if bad_h else "")

    check("prior_share within [0, 1]",
          hist.filter((pl.col("prior_share") < 0) | (pl.col("prior_share") > 1)).height == 0)

    # The time split must leave usable rows on both sides.
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    n_tr = v.filter(pl.col("ts") < cut).height
    n_te = v.filter(pl.col("ts") >= cut).height
    check("time split has rows on both sides", n_tr > 0 and n_te > 0,
          f"{n_tr:,} train / {n_te:,} test")


# --------------------------------------------------------------- models
def validate_models() -> None:
    print("\nMODELS")
    needed = ("model_category_relevance.joblib", "model_sku_affinity.joblib",
              "train_report.json")
    for f in needed:
        check(f"{f} exists", (C.ART / f).exists())
    if not all((C.ART / f).exists() for f in needed):
        sys.exit("\ntrain_models.py has not been run")

    rep = json.loads((C.ART / "train_report.json").read_text())

    for cat, st in rep["category_relevance"].items():
        check(f"relevance [{cat[:24]}] beats coin flip", st["oot_auc"] > 0.55,
              f"out-of-time AUC {st['oot_auc']:.3f}")
        check(f"relevance [{cat[:24]}] holds out of time",
              st["auc"] - st["oot_auc"] < 0.10,
              f"in-period {st['auc']:.3f} vs {st['oot_auc']:.3f}", warn_only=True)
        # Calibration is the whole job: these numbers are multiplied by
        # dollars, so a well-ranked but badly-scaled model is useless.
        gap = abs(st["oot_predicted"] - st["oot_actual"])
        check(f"relevance [{cat[:24]}] is calibrated", gap < 0.01,
              f"predicted {st['oot_predicted']:.4f} vs "
              f"actual {st['oot_actual']:.4f}")

    # SKU Affinity must not have seen the test period.
    lines = pl.read_parquet(C.ART / "line_clean.parquet",
                            columns=["ts", "category", "gtin"])
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    eligible_all = lines.filter(
        pl.col("category").is_in(C.SKU_MODEL_CATEGORIES)
        & pl.col("gtin").is_not_null()).height
    eligible_tr = lines.filter(
        pl.col("category").is_in(C.SKU_MODEL_CATEGORIES)
        & pl.col("gtin").is_not_null() & (pl.col("ts") < cut)).height
    check("sku affinity excludes the test period", eligible_tr < eligible_all,
          f"{eligible_tr:,} of {eligible_all:,} eligible lines")

    aff = rep["sku_affinity"]
    check("sku affinity matrix is not degenerate",
          aff["items"] > 50 and aff["shoppers"] > 1000,
          f"{aff['shoppers']:,} x {aff['items']:,}, {aff['nnz']:,} nonzero")
    mc = joblib.load(C.ART / "model_sku_affinity.joblib")
    uf = np.asarray(mc["als"].user_factors)
    itf = np.asarray(mc["als"].item_factors)
    check("ALS factor shapes line up",
          uf.shape == (aff["shoppers"], aff["factors"])
          and itf.shape == (aff["items"], aff["factors"]),
          f"user {uf.shape}, item {itf.shape}")
    check("ALS factors converged to finite values",
          bool(np.isfinite(uf).all() and np.isfinite(itf).all()))
    check("ALS factors are not all zero", float(np.abs(itf).max()) > 1e-6,
          f"max |item factor| {np.abs(itf).max():.4f}")


def validate_serving() -> None:
    print("\nSERVING")
    from serve import OfferEngine

    eng = OfferEngine()
    v = pl.read_parquet(C.ART / "visit.parquet", columns=["shopper_id"])
    known = v["shopper_id"][0]

    d = eng.decide(known, datetime(2026, 6, 15, 7, 30), "REGULAR")
    check("known shopper gets candidates", len(d.ranked) > 0,
          f"{len(d.ranked)} ranked")
    check("known shopper scores in range",
          all(0 < o.p_category < 1 for o in d.ranked))

    u = eng.decide("0" * 32, datetime(2026, 6, 15, 2, 15), "DIESEL")
    check("unseen shopper does not crash", u.prior_attach == -1.0,
          "no history, fell back to context plus popularity")
    # A hard zero means the isotonic calibrator's lowest step collapsed and
    # every expected value downstream would be zero too.
    check("no probability collapses to exactly zero",
          all(o.p_category > 0 for o in u.ranked))

    # Context must actually move the answer, or the model is a constant.
    def top_pcat(hour):
        r = eng.decide(known, datetime(2026, 6, 15, hour, 0), "REGULAR").ranked
        return r[0].p_category if r else 0.0
    am, night = top_pcat(6), top_pcat(2)
    check("time of day changes the prediction", abs(am - night) > 0.001,
          f"6am {am:.4f} vs 2am {night:.4f}")

    # Ranked on probability, not on expected value. The margins behind EV are
    # average price times an assumed 40%, so ranking on them would let an
    # assumption choose the SKU. EV rides along for negotiate.py.
    pb = [o.p_buy for o in d.ranked]
    check("offers are ranked by P(buys this SKU)", pb == sorted(pb, reverse=True))
    # No check that the EV order DIFFERS: for a candidate set with similar
    # margins the two orders legitimately coincide, and asserting otherwise
    # would be testing a coincidence rather than a property.
    check("funding share drives our cost",
          all(abs(o.our_cost - o.discount * (1 - o.funded_share)) < 1e-9
              for o in d.ranked))
    check("fully funded offers cost us nothing",
          all(o.our_cost == 0 for o in d.ranked if o.funded_share >= 0.999))
    # The negotiating floor must fall as the assumed uplift rises, or the
    # inversion is wrong and the number is unusable in a brand conversation.
    if d.ranked:
        o = d.ranked[0]
        lo, hi = o.min_funded_share(0.10), o.min_funded_share(1.00)
        check("minimum funding share falls as uplift rises", hi < lo,
              f"{lo:.0%} at +10%, {hi:.0%} at +100%")


def validate_batch_agreement() -> None:
    """The batch table and the live engine must give the same answer.

    They are two ways into `scoring.py` now, but they were two copies of the
    ranking before, and the copies drifted twice: once the batch path lost
    the within-category normalisation, once it lost P(category) out of the
    expected value. Both bugs were invisible until someone read the table and
    noticed every top slot was beer. This check is the tripwire.
    """
    print("\n-- batch vs live scoring --")
    from serve import OfferEngine, daypart_of
    import scoring
    from scoring import SkuRanker, slug

    eng = OfferEngine()
    rel = joblib.load(C.ART / "model_category_relevance.joblib")
    aff = joblib.load(C.ART / "model_sku_affinity.joblib")
    catalog = pl.read_csv(C.ART / "offer_catalog.csv",
                          schema_overrides={"gtin": pl.Utf8})

    visit = pl.read_parquet(C.ART / "visit.parquet")
    hist = pl.read_parquet(C.ART / "feat_history.parquet")
    wide = hist.pivot(on="category", index="fuel_tx",
                      values=["prior_n", "bought_before", "prior_share"],
                      aggregate_function="first")
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    # A handful of real test-period moments, batched the way score_table.py
    # batches them.
    sample = (visit.filter(pl.col("ts") >= cut)
              .join(wide, on="fuel_tx", how="left")
              .sample(12, seed=C.RANDOM_SEED))

    X = scoring.design_matrix(sample, rel["cat_maps"], rel["cols"])
    props = scoring.category_probs(rel["models"], X)
    ranker = SkuRanker(aff, catalog)
    batch = ranker.rank(sample["shopper_id"].to_list(),
                        sample["daypart"].to_list(), props, top=5)

    p_gap = ev_gap = 0.0
    same_pick = 0
    rows = sample.to_dicts()
    for i, r in enumerate(rows):
        live = eng.decide(r["shopper_id"], r["ts"], r["fuel_grade"],
                          store_id=r["store_id"], fuel_spend=r["fuel_spend"],
                          top=5)
        # Category probabilities, every category.
        for cat in props:
            p_gap = max(p_gap, abs(float(props[cat][i]) -
                                   eng.category_probs(
                                       r["shopper_id"], r["ts"],
                                       r["fuel_grade"],
                                       store_id=r["store_id"],
                                       fuel_spend=r["fuel_spend"])[cat]))
        b = batch[i]
        if not b and not live.ranked:
            same_pick += 1
            continue
        if b and live.ranked and b[0]["gtin"] == live.ranked[0].gtin:
            same_pick += 1
        for cand, off in zip(b, live.ranked):
            ev_gap = max(ev_gap, abs(cand["expected_value"] - off.expected_value))

    check("live and batch agree on P(category)", p_gap < 1e-9,
          f"largest gap {p_gap:.2e} across {len(rows)} fills x {len(props)} categories")
    check("live and batch agree on expected value", ev_gap < 1e-9,
          f"largest gap {ev_gap:.2e}")
    check("live and batch pick the same SKU", same_pick == len(rows),
          f"{same_pick}/{len(rows)} fills")

    # ---- the two-stage rule -------------------------------------------
    # Stage one picks a category; stage two may only look inside it. If a
    # SKU from a second category can reach the list, margin has started
    # choosing the category again -- the bug that offered candy to a
    # customer we were 90.6% sure wanted a beverage.
    chosen = ranker.choose_category(sample["daypart"].to_list(), props)
    one_cat = all(len({c["category"] for c in b}) <= 1 for b in batch)
    check("every ranked SKU comes from one category", one_cat,
          "stage two cannot leave the category stage one chose")
    check("the ranked category is the category stage one chose",
          all(not b or b[0]["category"] == chosen[i]
              for i, b in enumerate(batch)))

    # And that category must be the most relevant one we are ALLOWED to
    # show -- not the most relevant one overall, since tobacco is excluded.
    catalog_showable = catalog.filter(pl.col("in_stock") & pl.col("offerable"))
    allowed = set(catalog_showable["category"].unique().to_list()) & set(props)
    worst = 0.0
    for i, cat in enumerate(chosen):
        if cat is None:
            continue
        best = max(float(props[c][i]) for c in allowed)
        worst = max(worst, best - float(props[cat][i]))
    check("stage one picks the most relevant allowed category", worst < 1e-12,
          f"largest shortfall {worst:.2e} against the best allowed category")
    check("no excluded category is ever chosen",
          all(c not in C.NOT_OFFERABLE for c in chosen if c),
          f"excluded: {', '.join(C.NOT_OFFERABLE)}")

    # A customer with no signal at all sits at the floor in every category,
    # so the tie-break decides. It must land on the daypart's best seller,
    # not on whatever sorts first -- otherwise a brand-new customer is shown
    # beer because "Beer" begins with a B.
    flat = {c: np.array([scoring.P_FLOOR]) for c in props}
    for dp in ("morning", "midday", "evening"):
        got = ranker.choose_category([dp], flat)[0]
        pop = ranker.cat_popularity.get(dp, {})
        allowed_pop = {c: v for c, v in pop.items() if c in allowed}
        want = max(allowed_pop, key=lambda c: allowed_pop[c]) if allowed_pop else None
        check(f"no-signal customer falls back to the best seller ({dp})",
              got == want, f"{got}")

    # Inside one category P(category) is a constant multiplier, so ordering
    # by expected value and by p_sku x unit margin must agree.
    agree = all(
        all(b[j]["p_sku"] >= b[j + 1]["p_sku"] - 1e-12
            for j in range(len(b) - 1))
        for b in batch)
    check("within-category order is P(sku) alone", agree,
          "no assumed margin reorders the anchor prediction")

    # prob_entering is a reported number, so it has to be a probability and
    # it has to be at least as large as any single category it came from.
    entering = scoring.prob_entering(props)
    P = np.column_stack([props[k] for k in sorted(props)])
    check("prob_entering stays inside (0, 1)",
          bool(((entering > 0) & (entering < 1)).all()),
          f"range {entering.min():.4f} to {entering.max():.4f}")
    check("prob_entering is at least the largest single category",
          bool((entering >= P.max(axis=1) - 1e-12).all()))
    check("engine reports the same prob_entering",
          abs(eng.prob_entering(rows[0]["shopper_id"], rows[0]["ts"],
                                rows[0]["fuel_grade"],
                                store_id=rows[0]["store_id"],
                                fuel_spend=rows[0]["fuel_spend"])
              - float(entering[0])) < 1e-9)

    # The written table, if one exists, must match today's models.
    path = C.ART / "customer_scores.parquet"
    if not path.exists():
        check("customer_scores.parquet exists", False,
              "run score_table.py", warn_only=True)
        return
    tbl = pl.read_parquet(path)
    cols = set(tbl.columns)
    want = {f"{slug(c)}_propensity" for c in props}
    check("scored table carries one propensity column per category",
          want <= cols, f"{len(want)} categories")
    need = {"prob_entering", "offer_category", "top_skus"}
    check("scored table has prob_entering, offer_category and top_skus",
          need <= cols, "" if need <= cols
          else f"missing {', '.join(sorted(need - cols))} — rerun score_table.py")
    if not need <= cols:
        return
    mixed = (tbl.head(5_000)
             .with_columns(n_cat=pl.col("top_skus").list.eval(
                 pl.element().struct.field("category")).list.n_unique())
             .filter(pl.col("n_cat") > 1).height)
    check("written rows never mix categories in top_skus", mixed == 0,
          f"{mixed} of the first 5,000 rows mix")
    off = (tbl.head(5_000)
           .with_columns(first=pl.col("top_skus").list.first()
                         .struct.field("category"))
           .filter(pl.col("first") != pl.col("offer_category")).height)
    check("offer_category matches the SKUs beneath it", off == 0,
          f"{off} of the first 5,000 rows disagree")
    check("no reported probability rounds to a flat 0 or 1",
          bool(((tbl["prob_entering"] > 0) & (tbl["prob_entering"] < 1)).all()),
          f"prob_entering max {tbl['prob_entering'].max():.5f}")
    first = tbl["top_skus"][0].struct.unnest()
    check("top_skus carries all four probability fields",
          {"affinity", "p_sku", "p_category", "p_buys"} <= set(first.columns),
          "every term in the chain is auditable")
    check("top_skus carries no invented dollar figures",
          not ({"unit_margin", "funded_share", "expected_value"}
               & set(first.columns)),
          "margins are an assumed 40%, so they stay out of the table")
    # The struct is stored rounded to six decimals, so the identity only
    # holds to about that -- tighter than this would be testing the rounding.
    gap_pb = float((((first["p_category"] * first["p_sku"]) - first["p_buys"])
                    .abs()).max())
    check("p_buys is p_category x p_sku", gap_pb < 2e-6, f"gap {gap_pb:.2e}")
    check("top_skus is ranked by p_buys",
          first["p_buys"].to_list() ==
          sorted(first["p_buys"].to_list(), reverse=True))

    # ---- the second item ------------------------------------------------
    check("scored table has second_category and second_category_skus",
          {"second_category", "second_category_skus"} <= set(tbl.columns))
    head = tbl.head(5_000)
    with_pair = head.filter(pl.col("second_category_skus").list.len() > 0)
    check("most rows carry a pair offer",
          with_pair.height > head.height * 0.5,
          f"{with_pair.height:,} of {head.height:,} rows")
    check("the pair category is never the anchor category",
          head.filter(pl.col("second_category")
                      == pl.col("offer_category")).height == 0,
          "never discount the item they came for")
    sec_cats = set(head["second_category"].drop_nulls().to_list())
    check("no excluded category is ever the second category",
          not (sec_cats & set(C.NOT_OFFERABLE + C.NOT_PAIRABLE)),
          f"{len(sec_cats)} distinct pair categories")
    # Every SKU in the struct must sit inside second_category, for the same
    # reason top_skus must sit inside offer_category: the category is chosen
    # first, and the SKU may not wander out of it.
    mixed = with_pair.select(
        n=pl.col("second_category_skus").list.eval(
            pl.element().struct.field("category")).list.unique().list.len()
    ).filter(pl.col("n") > 1).height
    check("second_category_skus never mixes categories", mixed == 0,
          f"{mixed} rows mix")
    agree = with_pair.select(
        ok=pl.col("second_category_skus").list.first()
        .struct.field("category") == pl.col("second_category")
    ).filter(~pl.col("ok")).height
    check("second_category matches the SKUs beneath it", agree == 0,
          f"{agree} rows disagree")
    sfirst = with_pair["second_category_skus"][0].struct.unnest()
    check("second_category_skus carries the pairing evidence",
          {"cohort_index", "excess", "headroom"} <= set(sfirst.columns),
          "a ratio to gate on and a difference to rank on")
    check("second_category_skus claims no P(category) it does not have",
          not ({"p_category", "p_buys", "unit_margin", "expected_value"}
               & set(sfirst.columns)),
          "the pair category has no relevance model")
    check("second_category_skus is ranked by p_sku",
          sfirst["p_sku"].to_list()
          == sorted(sfirst["p_sku"].to_list(), reverse=True))
    check("every pair in the table clears the complement gate",
          bool((with_pair["second_category_skus"].list.first()
                .struct.field("cohort_index") >= C.PAIR_MIN_INDEX).all()),
          f"index >= {C.PAIR_MIN_INDEX}")
    alc = head.filter(pl.col("offer_category").is_in(C.NOT_PAIRABLE))
    check("an alcohol anchor carries no second item",
          alc.filter(pl.col("second_category_skus").list.len() > 0).height == 0,
          f"{alc.height:,} alcohol-anchored rows, all empty")

    # Spot-check one written row against the live engine.
    r = tbl.row(0, named=True)
    v = visit.filter((pl.col("shopper_id") == r["shopper_id"])
                     & (pl.col("ts") == r["ts"]))
    if v.height:
        vr = v.row(0, named=True)
        live = eng.category_probs(r["shopper_id"], r["ts"], r["fuel_grade"],
                                  store_id=r["store_id"],
                                  fuel_spend=vr["fuel_spend"])
        gap = max(abs(live[c] - r[f"{slug(c)}_propensity"]) for c in live)
        check("written table still matches the fitted models", gap < 1e-4,
              f"largest gap {gap:.2e}; a bigger one means the table is stale")


def pe_pairable_cats(t: pl.DataFrame) -> list[str]:
    """Pair categories that survive the gates the engine applies."""
    return (t.filter((pl.col("cohort_support") >= C.PAIR_MIN_SUPPORT)
                     & (pl.col("cohort_index") >= C.PAIR_MIN_INDEX)
                     & ~pl.col("pair").is_in(C.NOT_OFFERABLE
                                             + C.NOT_PAIRABLE))
             ["pair"].unique().to_list())


def validate_pairing() -> None:
    """The pair table and the pair offer.

    Pairing answers a different question from the anchor pipeline: not what
    they will buy, but what second thing a discount might add. The checks
    here are mostly about not fooling ourselves -- a ratio on a tiny base, a
    cohort that indexes high only because it is a cohort of heavy shoppers.
    """
    print("\n-- pairing --")
    path = C.ART / "pair_affinity.parquet"
    if not path.exists():
        check("pair_affinity.parquet exists", False,
              "run: python pairing.py --build", warn_only=True)
        return
    t = pl.read_parquet(path)
    cols = set(t.columns)
    want = {"anchor", "pair", "n_both", "confidence", "basket_lift",
            "cohort_rate", "other_rate", "cohort_index", "cohort_support"}
    check("pair table has the expected columns", want <= cols,
          "" if want <= cols else f"missing {', '.join(sorted(want - cols))}")
    if not want <= cols:
        return

    check("no category is paired with itself",
          t.filter(pl.col("anchor") == pl.col("pair")).height == 0)

    # Basket lift is symmetric by construction: P(A,B) / (P(A) P(B)). If the
    # two directions disagree, the marginals are being joined wrongly.
    sym = (t.select("anchor", "pair", "basket_lift")
           .join(t.select(pl.col("pair").alias("anchor"),
                          pl.col("anchor").alias("pair"),
                          pl.col("basket_lift").alias("rev")),
                 on=["anchor", "pair"], how="inner")
           .drop_nulls())
    gap = float((sym["basket_lift"] - sym["rev"]).abs().max() or 0.0)
    check("basket lift is symmetric", gap < 1e-9,
          f"largest gap {gap:.2e} over {sym.height:,} pairs")

    good = t.filter(pl.col("cohort_index").is_not_null())
    check("cohort rates are real probabilities",
          bool(((good["cohort_rate"] >= 0) & (good["cohort_rate"] <= 1)
                & (good["other_rate"] >= 0) & (good["other_rate"] <= 1)).all()))
    # The index and the excess must tell the same story about direction, or
    # one of them is computed off the wrong denominator.
    disagree = good.filter(
        ((pl.col("cohort_index") > 1) & (pl.col("cohort_rate")
                                         < pl.col("other_rate")))
        | ((pl.col("cohort_index") < 1) & (pl.col("cohort_rate")
                                           > pl.col("other_rate")))).height
    check("index and excess agree on direction", disagree == 0,
          f"{disagree} pairs disagree")

    # The mission structure, as a regression test. These are the findings the
    # write-ups quote, so they should fail loudly if the data or the cohort
    # definition shifts under them.
    def idx(a, b):
        r = t.filter((pl.col("anchor") == a) & (pl.col("pair") == b))
        return float(r["cohort_index"][0]) if r.height else float("nan")

    bev = "Packaged Beverages (Non-Alcoholic)"
    check("snack mission: drinks pair with alternative snacks",
          idx(bev, "Alternative Snacks") > 1.5,
          f"index {idx(bev, 'Alternative Snacks'):.2f}")
    check("drinks and candy are NOT complements",
          idx(bev, "Candy") < C.PAIR_MIN_INDEX + 0.05,
          f"index {idx(bev, 'Candy'):.2f} — popular, not complementary")
    check("beer and drinks are substitutes, not complements",
          idx("Beer", bev) < 0.5, f"index {idx('Beer', bev):.2f}")
    # Alcohol is excluded from pairing on both sides as a business rule. The
    # measurement agrees -- beer's only complement above 1.0 is Liquor at
    # 1.08 -- but the rule does not depend on the measurement holding.
    check("alcohol never appears as a pair candidate",
          all(c not in C.NOT_PAIRABLE
              for c in pe_pairable_cats(t) if c),
          f"excluded: {', '.join(C.NOT_PAIRABLE)}")
    check("cigarettes pair with almost nothing",
          t.filter((pl.col("anchor") == "Cigarettes")
                   & (pl.col("cohort_index") >= C.PAIR_MIN_INDEX)
                   & (pl.col("cohort_support") >= C.PAIR_MIN_SUPPORT)
                   ).height <= 2,
          "a tobacco run is its own mission")

    # And the live recommendation.
    from pairing import PairEngine
    pe = PairEngine()
    busy = (pe.engine.visit.group_by("shopper_id")
            .agg(n=pl.len(), k=pl.col("bought").sum())
            .filter((pl.col("n") >= 8) & (pl.col("k") >= 5))
            .sort(["k", "shopper_id"], descending=[True, False]).row(0)[0])
    offers, anchor = pe.recommend(busy, datetime(2026, 6, 15, 7, 30), "REGULAR")
    check("pair offer returns candidates", len(offers) > 0, f"{len(offers)}")
    check("the pair is never the anchor category",
          all(o.pair_category != anchor.category for o in offers))
    check("pair candidates clear the complement gate",
          all(o.cohort_index >= C.PAIR_MIN_INDEX for o in offers))
    check("no excluded category is ever paired",
          all(o.pair_category not in C.NOT_OFFERABLE + C.NOT_PAIRABLE
              for o in offers))
    # The three strongest complements to a drink used to name a category and
    # then shrug, because TARGET_CATEGORIES was chosen for the anchor
    # question. They are in the catalog now, so a pair offer should be
    # showable rather than hypothetical.
    check("the top pair resolves to a real SKU", offers[0].in_catalog
          and offers[0].pair_gtin is not None,
          f"{offers[0].pair_category} -> {offers[0].pair_brand}")
    in_cat = sum(o.in_catalog for o in offers)
    check("most pair candidates have SKUs we could show",
          in_cat >= max(1, len(offers) - 1),
          f"{in_cat} of {len(offers)}")
    check("pair offers are ranked by excess x headroom",
          [o.score for o in offers] == sorted((o.score for o in offers),
                                              reverse=True))
    check("excess is cohort rate minus the comparison rate",
          all(abs(o.excess - (o.cohort_rate - o.other_rate)) < 1e-12
              for o in offers))
    # Not a check -- the two orderings genuinely coincide for some anchors.
    # Printed because when they diverge, that divergence IS the finding.
    by_rate = max(offers, key=lambda o: o.cohort_rate)
    by_index = max(offers, key=lambda o: o.cohort_index)
    print(f"  [note ] top pair by excess {offers[0].pair_category[:24]!r}; "
          f"by raw rate {by_rate.pair_category[:24]!r}; "
          f"by index {by_index.pair_category[:24]!r}")
    # The batch column and the live engine pick the pair through two
    # different code paths (BatchPairer.choose and PairEngine.recommend).
    # They share the gates via pair_candidates, but not the selection loop,
    # and this pipeline has twice shipped a batch table that disagreed with
    # the live answer. So compare them on real written rows.
    spath = C.ART / "customer_scores.parquet"
    if spath.exists():
        rows = (pl.read_parquet(spath)
                .filter(pl.col("second_category_skus").list.len() > 0)
                .head(400).sample(min(8, 400), seed=C.RANDOM_SEED))
        same = diff = 0
        for r in rows.iter_rows(named=True):
            live, _ = pe.recommend(r["shopper_id"], r["ts"], r["fuel_grade"],
                                   store_id=int(r["store_id"]))
            got = live[0].pair_category if live else None
            same += got == r["second_category"]
            diff += got != r["second_category"]
        check("the batch pair offer matches the live one",
              diff == 0, f"{same} agree, {diff} disagree over {rows.height} rows")
    else:
        check("customer_scores.parquet exists to compare against", False,
              "run score_table.py", warn_only=True)

    # A beer anchor is the case the rule exists for: no second item at all,
    # which is a real answer rather than a failure to find one.
    beer_offers = pe.recommend(busy, datetime(2026, 6, 15, 19, 30), "REGULAR",
                               force_anchor="Beer")[0]
    check("a beer anchor returns no pair offer", beer_offers == [],
          "a beer run is its own mission")
    if anchor.ranked:
        check("the anchor SKU is excluded from the pair",
              all(o.pair_gtin != anchor.ranked[0].gtin for o in offers),
              "never discount the item they came for")


def validate_sku_affinity() -> None:
    """Stage two, judged on whether it named the item they bought.

    Calibration checks stage one and says nothing about stage two. This is
    the hit@k check, scored on the SAME rows for both rankings so the
    comparison is not just "shoppers with history buy more".

    Kept light here -- validate_sku.py prints the full breakdown.
    """
    print("\n-- SKU affinity --")
    try:
        from validate_sku import basket_gtins, popularity_topk
    except Exception as e:                       # pragma: no cover
        check("validate_sku imports", False, str(e), warn_only=True)
        return
    spath = C.ART / "customer_scores.parquet"
    if not spath.exists():
        check("customer_scores.parquet exists", False, "run score_table.py",
              warn_only=True)
        return
    from scoring import SkuRanker
    aff = joblib.load(C.ART / "model_sku_affinity.joblib")
    catalog = pl.read_csv(C.ART / "offer_catalog.csv",
                          schema_overrides={"gtin": pl.Utf8})
    ranker = SkuRanker(aff, catalog)
    scores = pl.read_parquet(spath, columns=[
        "fuel_tx", "daypart", "offer_category", "personalised", "top_skus"])
    df = (scores.join(basket_gtins(), on="fuel_tx", how="inner")
          .filter(pl.col("bought_cats")
                  .list.contains(pl.col("offer_category"))))
    check("there are fills to judge stage two on", df.height > 1000,
          f"{df.height:,} fills bought inside the offered category")
    if df.height == 0:
        return
    cache, hit, pop, pers = {}, [], [], []
    for r in df.select("daypart", "offer_category", "personalised",
                       "top_skus", "gtins").to_dicts():
        truth = set(r["gtins"])
        key = (r["daypart"], r["offer_category"])
        if key not in cache:
            cache[key] = popularity_topk(ranker, *key, 10)
        hit.append(bool(truth & {c["gtin"] for c in r["top_skus"][:10]}))
        pop.append(bool(truth & set(cache[key][:10])))
        pers.append(bool(r["personalised"]))
    hit, pop, pers = np.array(hit), np.array(pop), np.array(pers)

    a, b = hit[pers].mean(), pop[pers].mean()
    check("SKU Affinity beats the best seller where it runs", a > b * 1.3,
          f"hit@10 {a:.1%} vs {b:.1%} — {a / b:.2f}x on the same fills")
    # The fallback path IS the popularity ranking, so the two must agree
    # exactly there. If they ever diverge, the fallback has drifted.
    same = (hit[~pers] == pop[~pers]).mean() if (~pers).sum() else 1.0
    check("the fallback path is exactly the best-seller ranking", same > 0.999,
          f"{same:.3%} of fallback rows identical")
    check("the personalised share matches the published figure",
          abs(float(scores["personalised"].mean()) - 0.316) < 0.03,
          f"{float(scores['personalised'].mean()):.1%} of scored fills")


def main() -> None:
    print("=" * 66)
    print("pump-screen pipeline validation")
    print("=" * 66)
    v = validate_tables()
    validate_leakage(v)
    validate_models()
    validate_serving()
    validate_batch_agreement()
    validate_pairing()
    validate_sku_affinity()

    print("\n" + "=" * 66)
    print(f"{len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed")
    for w in WARN:
        print(f"  warn: {w}")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
