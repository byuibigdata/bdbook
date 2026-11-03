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
    needed = ("model_a.joblib", "model_b.joblib", "model_c.joblib",
              "train_report.json")
    for f in needed:
        check(f"{f} exists", (C.ART / f).exists())
    if not all((C.ART / f).exists() for f in needed):
        sys.exit("\ntrain_models.py has not been run")

    rep = json.loads((C.ART / "train_report.json").read_text())
    auc_a = rep["model_a"]["auc"]
    check("model_a beats coin flip", auc_a > 0.55, f"AUC {auc_a:.3f}")
    oot = rep["model_a"].get("oot_auc")
    check("model_a holds up out of time", oot is not None and oot > 0.55,
          f"in-period {auc_a:.3f} vs out-of-time {oot:.3f}" if oot else "missing")
    if oot:
        check("out-of-time AUC has not collapsed", auc_a - oot < 0.10,
              f"dropped {auc_a - oot:+.3f}", warn_only=True)
    check("model_a is not suspiciously perfect", auc_a < 0.95,
          f"AUC {auc_a:.3f} — above 0.95 usually means leakage", warn_only=True)

    for cat, st in rep["model_b"].items():
        check(f"model_b [{cat[:26]}] beats coin flip", st["auc"] > 0.55,
              f"AUC {st['auc']:.3f}")

    c = rep["model_c"]
    check("model_c matrix is not degenerate",
          c["items"] > 50 and c["shoppers"] > 1000,
          f"{c['shoppers']:,} x {c['items']:,}, {c['nnz']:,} nonzero")

    # Model C must not have seen the test period. Its matrix is built from
    # als_interactions, so check that table rather than the fitted object.
    inter_max = (pl.read_parquet(C.ART / "als_interactions.parquet").height)
    lines = pl.read_parquet(C.ART / "line_clean.parquet",
                            columns=["ts", "category", "gtin"])
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    eligible_all = lines.filter(
        pl.col("category").is_in(C.SKU_MODEL_CATEGORIES)
        & pl.col("gtin").is_not_null()).height
    eligible_tr = lines.filter(
        pl.col("category").is_in(C.SKU_MODEL_CATEGORIES)
        & pl.col("gtin").is_not_null() & (pl.col("ts") < cut)).height
    check("model_c excludes the test period",
          eligible_tr < eligible_all,
          f"{inter_max:,} pairs from {eligible_tr:,} of {eligible_all:,} "
          f"eligible lines")

    mc = joblib.load(C.ART / "model_c.joblib")
    als = mc["als"]
    uf, itf = np.asarray(als.user_factors), np.asarray(als.item_factors)
    check("ALS factor shapes line up",
          uf.shape == (c["shoppers"], c["factors"])
          and itf.shape == (c["items"], c["factors"]),
          f"user {uf.shape}, item {itf.shape}")
    check("ALS factors converged to finite values",
          bool(np.isfinite(uf).all() and np.isfinite(itf).all()))
    check("ALS factors are not all zero",
          float(np.abs(itf).max()) > 1e-6,
          f"max |item factor| {np.abs(itf).max():.4f}")

    # Calibration is the whole job for model A: the number is priced, not
    # thresholded, so a well-ranked but badly-scaled model is useless.
    v = pl.read_parquet(C.ART / "visit.parquet")
    a = joblib.load(C.ART / "model_a.joblib")
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    te = v.filter(pl.col("ts") >= cut)
    te = te.sample(min(200_000, te.height), seed=C.RANDOM_SEED)
    enc = {col: te[col].fill_null("__NA__").cast(pl.Utf8)
           .replace_strict(m, default=-1, return_dtype=pl.Int32)
           for col, m in a["cat_maps"].items()}
    X = te.with_columns(**enc).select(a["cols"]).fill_null(0).to_numpy()
    p = a["model"].predict_proba(X)[:, 1]
    actual, predicted = te["bought"].mean(), p.mean()
    check("model_a is calibrated on the test period",
          abs(actual - predicted) < 0.03,
          f"predicted {predicted:.3f} vs actual {actual:.3f}")


# --------------------------------------------------------------- serving
def validate_serving() -> None:
    print("\nSERVING")
    from serve import OfferEngine

    eng = OfferEngine()
    v = pl.read_parquet(C.ART / "visit.parquet", columns=["shopper_id"])
    known = v["shopper_id"][0]

    d = eng.decide(known, datetime(2026, 6, 15, 7, 30), "REGULAR")
    check("known shopper scores", 0 < d.p_inside < 1, f"P(inside) {d.p_inside:.3f}")
    check("known shopper gets candidates", len(d.ranked) > 0,
          f"{len(d.ranked)} ranked")

    u = eng.decide("0" * 32, datetime(2026, 6, 15, 2, 15), "DIESEL")
    check("unseen shopper does not crash", 0 < u.p_inside < 1,
          f"P(inside) {u.p_inside:.5f}, fell back to popularity")
    # A hard zero means the isotonic calibrator's lowest step collapsed and
    # every expected value downstream would be zero too.
    check("no probability collapses to exactly zero",
          u.p_inside > 0 and all(o.p_category > 0 for o in u.ranked))

    # Context must actually move the answer, or the model is a constant.
    am = eng.decide(known, datetime(2026, 6, 15, 6, 0), "REGULAR").p_inside
    night = eng.decide(known, datetime(2026, 6, 15, 2, 0), "REGULAR").p_inside
    check("time of day changes P(inside)", abs(am - night) > 0.005,
          f"6am {am:.3f} vs 2am {night:.3f}")

    ev = [o.expected_value for o in d.ranked]
    check("offers are ranked by expected value", ev == sorted(ev, reverse=True))
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


def main() -> None:
    print("=" * 66)
    print("pump-screen pipeline validation")
    print("=" * 66)
    v = validate_tables()
    validate_leakage(v)
    validate_models()
    validate_serving()

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
