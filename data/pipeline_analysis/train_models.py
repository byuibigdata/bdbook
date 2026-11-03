"""Fit the three models with scikit-learn, and save them for serving.

    model_a.joblib   will they come inside?      calibrated binary GBT
    model_b.joblib   which category, given in?   six binary GBTs
    model_c.joblib   which SKU in the category?  implicit-feedback ALS

Everything is split on time, never at random: a random split lets a model
learn from next month to predict last month, and the score it reports is a
number nobody can collect on.

Run:  uv run --with polars --with scikit-learn python train_models.py
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

import config as C

CTX_NUM = ["hour", "weekday", "month", "fuel_spend"]
# Model A gets the shopper's own prior attach rate. An earlier version was
# context-only, justified by a table bucketed on prior visit COUNT -- the
# wrong variable. Count is worth +0.004 AUC; the rate is worth +0.109.
A_HIST = ["prior_n", "prior_attach"]
CTX_CAT = ["daypart", "fuel_grade", "store_id"]
HIST_NUM = ["prior_n", "bought_before", "prior_share"]


def log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    print(f"[train] {msg}" + (f"   ({now - t0:.1f}s)" if t0 else ""), flush=True)
    return now


def encode(df: pl.DataFrame, cols_cat: list[str], maps: dict | None = None):
    """Integer-encode categoricals so HistGradientBoosting can use them natively.

    Returns the matrix and the fitted maps, so serving encodes identically.
    Unseen levels become -1, which the estimator treats as its own category.
    """
    fit = maps is None
    maps = {} if fit else maps
    out = {}
    for c in cols_cat:
        vals = df[c].fill_null("__NA__").cast(pl.Utf8)
        if fit:
            maps[c] = {v: i for i, v in enumerate(sorted(vals.unique().to_list()))}
        out[c] = vals.replace_strict(maps[c], default=-1, return_dtype=pl.Int32)
    return df.with_columns(**out), maps


def fit_binary(X, y, Xc, yc, label: str) -> tuple:
    """Fit a GBT, then calibrate it on held-out rows.

    No resampling. The output feeds an arithmetic in dollars, so a model
    trained on a balanced sample would score an average visit at 0.5 instead
    of its true base rate and inflate every expected value.
    """
    base = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=6,
        categorical_features=[i for i, _ in enumerate(CTX_CAT)],
        early_stopping=True,
        validation_fraction=0.1,
        random_state=C.RANDOM_SEED,
    )
    base.fit(X, y)
    # Isotonic needs enough rows per bin; fall back to sigmoid when thin.
    method = "isotonic" if len(yc) > 20_000 else "sigmoid"
    # sklearn >= 1.8 replaced cv="prefit" with an explicit frozen wrapper.
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    cal.fit(Xc, yc)
    p = cal.predict_proba(Xc)[:, 1]
    auc = roc_auc_score(yc, p) if len(set(yc)) > 1 else float("nan")
    print(f"        {label:<34} base {y.mean():6.3%}  "
          f"AUC {auc:.3f}  Brier {brier_score_loss(yc, p):.4f}")
    return cal, {"auc": float(auc), "base_rate": float(y.mean())}


def main() -> None:
    t = log("loading built tables")
    visit = pl.read_parquet(C.ART / "visit.parquet")
    hist = pl.read_parquet(C.ART / "feat_history.parquet")
    t = log(f"visit {visit.height:,}  feat_history {hist.height:,}", t)

    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    visit = visit.with_columns(is_train=pl.col("ts") < cut)
    n_tr, n_te = visit["is_train"].sum(), (~visit["is_train"]).sum()
    log(f"time split at {C.TRAIN_CUTOFF}: {n_tr:,} train / {n_te:,} test")
    if n_tr == 0 or n_te == 0:
        raise SystemExit("time split produced an empty side; check TRAIN_CUTOFF")

    # ---------------------------------------------------------------- A
    t = log("Model A - will they come inside?")
    a = visit.filter("is_train")
    if a.height > C.SAMPLE_VISITS:
        a = a.sample(C.SAMPLE_VISITS, seed=C.RANDOM_SEED)
    a, cat_maps = encode(a, CTX_CAT)
    a = a.sample(fraction=1.0, shuffle=True, seed=C.RANDOM_SEED)
    k = int(a.height * (1 - C.CALIB_FRACTION))
    cols = CTX_CAT + CTX_NUM + A_HIST              # categoricals first
    Xa = a.select(cols).fill_null(0).to_numpy()
    ya = a["bought"].cast(pl.Int8).to_numpy()
    model_a, stats_a = fit_binary(Xa[:k], ya[:k], Xa[k:], ya[k:], "model_a")

    # The numbers above are out-of-SAMPLE but still in-PERIOD. The honest
    # test is the data after the cutoff, which the model has never seen at
    # any point in time.
    te = visit.filter(~pl.col("is_train"))
    te, _ = encode(te, CTX_CAT, cat_maps)
    Xt = te.select(cols).fill_null(0).to_numpy()
    yt = te["bought"].cast(pl.Int8).to_numpy()
    pt = model_a.predict_proba(Xt)[:, 1]
    stats_a["oot_auc"] = float(roc_auc_score(yt, pt))
    stats_a["oot_brier"] = float(brier_score_loss(yt, pt))
    stats_a["oot_predicted_rate"] = float(pt.mean())
    stats_a["oot_actual_rate"] = float(yt.mean())
    print(f"        {'model_a OUT OF TIME':<34} actual {yt.mean():6.3%}  "
          f"AUC {stats_a['oot_auc']:.3f}  Brier {stats_a['oot_brier']:.4f}  "
          f"predicted {pt.mean():.3%}")
    t = log("model_a fitted", t)

    # ---------------------------------------------------------------- B
    t = log("Model B - which category, given they came inside?")
    conv = visit.filter(pl.col("is_train") & pl.col("bought"))
    wide = (
        hist.join(conv.select("fuel_tx"), on="fuel_tx", how="inner")
        .pivot(on="category", index="fuel_tx",
               values=["prior_n", "bought_before", "prior_share"],
               aggregate_function="first")
    )
    b = conv.join(wide, on="fuel_tx", how="left")
    b, _ = encode(b, CTX_CAT, cat_maps)
    b = b.sample(fraction=1.0, shuffle=True, seed=C.RANDOM_SEED)
    kb = int(b.height * (1 - C.CALIB_FRACTION))
    hist_cols = [c for c in b.columns if c.startswith(tuple(HIST_NUM))]
    cols_b = CTX_CAT + CTX_NUM + hist_cols
    Xb = b.select(cols_b).fill_null(0).to_numpy()

    models_b, stats_b = {}, {}
    for cat in C.TARGET_CATEGORIES:
        y = b["cats"].list.contains(cat).fill_null(False).cast(pl.Int8).to_numpy()
        if y[:kb].sum() < 200:
            print(f"        {cat[:34]:<34} skipped - too few positives")
            continue
        m, st = fit_binary(Xb[:kb], y[:kb], Xb[kb:], y[kb:], cat[:34])
        models_b[cat], stats_b[cat] = m, st
    t = log(f"model_b fitted for {len(models_b)} categories", t)

    # ---------------------------------------------------------------- C
    t = log("Model C - which SKU, via implicit-feedback ALS")
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
    X = csr_matrix((vals, (rows, cols_)), shape=(len(shoppers), len(items)))

    k = min(C.ALS_FACTORS, min(X.shape) - 1)
    als = AlternatingLeastSquares(
        factors=k,
        regularization=C.ALS_REGULARIZATION,
        alpha=C.ALS_ALPHA,
        iterations=C.ALS_ITERATIONS,
        random_state=C.RANDOM_SEED,
    )
    als.fit(X, show_progress=False)
    popularity = np.asarray(X.sum(axis=0)).ravel()      # cold-start fallback
    t = log(f"model_c: {X.shape[0]:,} shoppers x {X.shape[1]:,} items, "
            f"{k} factors, {C.ALS_ITERATIONS} iterations, {X.nnz:,} nonzero", t)

    # ------------------------------------------------------------- save
    joblib.dump({"model": model_a, "cat_maps": cat_maps, "cols": cols}, C.ART / "model_a.joblib")
    joblib.dump({"models": models_b, "cols": cols_b, "cat_maps": cat_maps,
                 "hist_cols": hist_cols}, C.ART / "model_b.joblib")
    joblib.dump({"als": als, "shopper_ix": s_ix, "item_ix": i_ix,
                 "items": items, "popularity": popularity,
                 "matrix": X}, C.ART / "model_c.joblib")
    (C.ART / "train_report.json").write_text(json.dumps(
        {"cutoff": C.TRAIN_CUTOFF, "n_train": int(n_tr), "n_test": int(n_te),
         "model_a": stats_a, "model_b": stats_b,
         "model_c": {"shoppers": X.shape[0], "items": X.shape[1],
                     "factors": k, "iterations": C.ALS_ITERATIONS,
                     "nnz": int(X.nnz)}}, indent=2))
    log("saved. next: validate.py, then serve.py")


if __name__ == "__main__":
    main()
