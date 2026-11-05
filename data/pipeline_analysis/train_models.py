"""Fit the two models and save them for serving.

    model_category_relevance.joblib   which category, if any?   six binary GBTs
    model_sku_affinity.joblib         which SKU inside it?      ALS

There used to be a third -- an attach model answering "will they come inside
at all" -- fitted first and multiplied into the others. It was dropped after
measuring it: P(category) = P(inside) x P(category | inside) is an identity,
so two stages and one direct model score the same to three decimals. The
split cost an extra moving part and bought no accuracy.

Everything is split on time, never at random: a random split lets a model
learn from next month to predict last month, and the score it reports is a
number nobody can collect on.

Run:  uv run --with polars --with scikit-learn --with implicit python train_models.py
"""

from __future__ import annotations

import json
import re
import time

import joblib
import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

import config as C

# Categoricals first: HistGradientBoosting takes their positions by index.
CTX_CAT = ["daypart", "fuel_grade", "store_id"]
CTX_NUM = ["hour", "weekday", "month", "fuel_spend"]
# The shopper's own attach habit. How MANY times we have seen them is worth
# +0.004 AUC; what SHARE of those visits ended inside is worth +0.109.
SHOPPER = ["prior_n", "prior_attach"]
PER_CAT = ("prior_n_", "bought_before_", "prior_share_")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    print(f"[train] {msg}" + (f"   ({now - t0:.1f}s)" if t0 else ""), flush=True)
    return now


def encode(df: pl.DataFrame, cols: list[str], maps: dict | None = None):
    """Integer-encode categoricals, returning the maps so serving matches.

    Unseen levels become -1, which the estimator treats as its own category.
    """
    fit = maps is None
    maps = {} if fit else maps
    out = {}
    for c in cols:
        vals = df[c].fill_null("__NA__").cast(pl.Utf8)
        if fit:
            maps[c] = {v: i for i, v in enumerate(sorted(vals.unique().to_list()))}
        out[c] = vals.replace_strict(maps[c], default=-1, return_dtype=pl.Int32)
    return df.with_columns(**out), maps


def fit_calibrated(X, y, Xc, yc):
    """A gradient-boosted classifier, then calibrated on held-out rows.

    No resampling. The output is multiplied by dollars, so a model trained on
    a balanced sample would score an average visit at 0.5 instead of its true
    base rate and inflate every expected value.
    """
    base = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6,
        categorical_features=list(range(len(CTX_CAT))),
        early_stopping=True, validation_fraction=0.1,
        random_state=C.RANDOM_SEED,
    )
    base.fit(X, y)
    # sklearn >= 1.8 replaced cv="prefit" with an explicit frozen wrapper.
    method = "isotonic" if len(yc) > 20_000 else "sigmoid"
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    cal.fit(Xc, yc)
    return cal


def main() -> None:
    t = log("loading built tables")
    visit = pl.read_parquet(C.ART / "visit.parquet")
    hist = pl.read_parquet(C.ART / "feat_history.parquet")
    t = log(f"visit {visit.height:,}  feat_history {hist.height:,}", t)

    wide = hist.pivot(
        on="category", index="fuel_tx",
        values=["prior_n", "bought_before", "prior_share"],
        aggregate_function="first")
    visit = visit.join(wide, on="fuel_tx", how="left")

    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    visit = visit.with_columns(is_train=pl.col("ts") < cut)
    n_tr, n_te = visit["is_train"].sum(), (~visit["is_train"]).sum()
    log(f"time split at {C.TRAIN_CUTOFF}: {n_tr:,} train / {n_te:,} test")
    if n_tr == 0 or n_te == 0:
        raise SystemExit("time split produced an empty side; check TRAIN_CUTOFF")

    # ------------------------------------------- Category Relevance (GBT)
    t = log("Category Relevance — six binary classifiers, one per category")
    tr = visit.filter("is_train")
    if tr.height > C.SAMPLE_VISITS:
        tr = tr.sample(C.SAMPLE_VISITS, seed=C.RANDOM_SEED)
    tr, cat_maps = encode(tr, CTX_CAT)
    te, _ = encode(visit.filter(~pl.col("is_train")), CTX_CAT, cat_maps)

    per_cat = [c for c in visit.columns if c.startswith(PER_CAT)]
    cols = CTX_CAT + CTX_NUM + SHOPPER + per_cat
    tr = tr.sample(fraction=1.0, shuffle=True, seed=C.RANDOM_SEED)
    k = int(tr.height * (1 - C.CALIB_FRACTION))
    X = tr.select(cols).fill_null(0).to_numpy()
    Xte = te.select(cols).fill_null(0).to_numpy()

    models, stats = {}, {}
    print(f"        {'category':<34}{'base':>8}{'AUC':>7}{'oot AUC':>9}"
          f"{'oot Brier':>11}")
    for cat in C.TARGET_CATEGORIES:
        y = tr["cats"].list.contains(cat).fill_null(False).cast(pl.Int8).to_numpy()
        y_te = te["cats"].list.contains(cat).fill_null(False).cast(pl.Int8).to_numpy()
        if y[:k].sum() < 200 or len(set(y_te)) < 2:
            print(f"        {cat[:33]:<34}  too few positives — skipped")
            continue
        m = fit_calibrated(X[:k], y[:k], X[k:], y[k:])
        p_in = m.predict_proba(X[k:])[:, 1]
        p_oot = m.predict_proba(Xte)[:, 1]
        stats[cat] = {
            "base_rate": float(y.mean()),
            "auc": float(roc_auc_score(y[k:], p_in)),
            "oot_auc": float(roc_auc_score(y_te, p_oot)),
            "oot_brier": float(brier_score_loss(y_te, p_oot)),
            "oot_predicted": float(p_oot.mean()),
            "oot_actual": float(y_te.mean()),
        }
        models[cat] = m
        s = stats[cat]
        print(f"        {cat[:33]:<34}{s['base_rate']:>8.2%}{s['auc']:>7.3f}"
              f"{s['oot_auc']:>9.3f}{s['oot_brier']:>11.4f}")
    t = log(f"fitted {len(models)} category models", t)

    # ------------------------------------------------ SKU Affinity (ALS)
    t = log("SKU Affinity — implicit-feedback ALS")
    inter = pl.read_parquet(C.ART / "als_interactions.parquet")
    shoppers = inter["shopper_id"].unique().sort().to_list()
    items = inter["item"].unique().sort().to_list()
    s_ix = {s: i for i, s in enumerate(shoppers)}
    i_ix = {s: i for i, s in enumerate(items)}
    rows = inter["shopper_id"].replace_strict(s_ix).to_numpy()
    cols_ = inter["item"].replace_strict(i_ix).to_numpy()
    # Raw counts: ALS turns them into confidence itself via alpha, so
    # pre-damping them here would apply the transform twice.
    vals = inter["n"].to_numpy().astype(np.float32)
    Xals = csr_matrix((vals, (rows, cols_)), shape=(len(shoppers), len(items)))

    factors = min(C.ALS_FACTORS, min(Xals.shape) - 1)
    als = AlternatingLeastSquares(
        factors=factors, regularization=C.ALS_REGULARIZATION,
        alpha=C.ALS_ALPHA, iterations=C.ALS_ITERATIONS,
        random_state=C.RANDOM_SEED)
    als.fit(Xals, show_progress=False)
    popularity = np.asarray(Xals.sum(axis=0)).ravel()   # the fallback for 76%
    t = log(f"{Xals.shape[0]:,} shoppers x {Xals.shape[1]:,} items, "
            f"{factors} factors, {Xals.nnz:,} nonzero", t)

    # ------------------------------------------------------------- save
    joblib.dump({"models": models, "cols": cols, "cat_maps": cat_maps,
                 "per_cat": per_cat},
                C.ART / "model_category_relevance.joblib")
    joblib.dump({"als": als, "shopper_ix": s_ix, "item_ix": i_ix,
                 "items": items, "popularity": popularity, "matrix": Xals},
                C.ART / "model_sku_affinity.joblib")
    (C.ART / "train_report.json").write_text(json.dumps({
        "cutoff": C.TRAIN_CUTOFF, "n_train": int(n_tr), "n_test": int(n_te),
        "category_relevance": stats,
        "sku_affinity": {"shoppers": Xals.shape[0], "items": Xals.shape[1],
                         "factors": factors, "iterations": C.ALS_ITERATIONS,
                         "nnz": int(Xals.nnz)}}, indent=2))
    log("saved. next: validate.py, then serve.py")


if __name__ == "__main__":
    main()
