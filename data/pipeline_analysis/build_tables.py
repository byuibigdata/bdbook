"""Build every table the models need, with Polars.

Reads the five bronze PDI folders and writes parquet artifacts:

    line_clean.parquet        one row per sellable line
    visit.parquet             one row per fuel visit  <- the label
    feat_history.parquet      shopper x category, point-in-time
    als_interactions.parquet  shopper x sku@daypart counts
    offer_catalog.csv         the candidate offers (margins are ASSUMED)

Run:  uv run --with polars python build_tables.py
      uv run --with polars python build_tables.py --stores 120   # a subset
"""

from __future__ import annotations

import argparse
import time

import polars as pl

import config as C


def log(msg: str, t0: float | None = None) -> float:
    now = time.time()
    suffix = f"   ({now - t0:.1f}s)" if t0 else ""
    print(f"[build] {msg}{suffix}", flush=True)
    return now


def daypart_expr(ts: pl.Expr) -> pl.Expr:
    h = ts.dt.hour()
    return (
        pl.when(h.is_between(5, 9)).then(pl.lit("morning"))
        .when(h.is_between(10, 14)).then(pl.lit("midday"))
        .otherwise(pl.lit("evening"))
    )


def build_line_clean(store_limit: int | None) -> pl.DataFrame:
    """Clean, categorised, shopper-attributed line items."""
    gtin = (
        pl.scan_parquet(C.SRC["gtin"])
        .select("GTIN", "CATEGORY", "BRAND")
        .unique(subset="GTIN")
    )
    shopper = (
        pl.scan_parquet(C.SRC["shopper"])
        .select("TRANSACTION_SET_ID", "SHOPPER_ID")
        .unique(subset="TRANSACTION_SET_ID")  # split tender -> duplicate rows
    )
    lf = (
        pl.scan_parquet(C.SRC["items"])
        .with_columns(
            amount=pl.col("GRAND_TOTAL_AMOUNT").cast(pl.Float64),
            qty=pl.col("UNIT_QUANTITY").cast(pl.Float64),
            store_id=pl.col("STORE_ID").cast(pl.Int64),
        )
        .filter((pl.col("qty") > 0) & (pl.col("amount") > 0))
        .join(gtin, on="GTIN", how="left")
        .with_columns(category=pl.coalesce("CATEGORY", "NONSCAN_CATEGORY"))
        .drop_nulls("category")
        .join(shopper, on="TRANSACTION_SET_ID", how="inner")
        .select(
            tx=pl.col("TRANSACTION_SET_ID").cast(pl.Utf8),
            shopper_id="SHOPPER_ID",
            store_id="store_id",
            ts="DATE_TIME",
            category="category",
            brand="BRAND",
            gtin="GTIN",
            amount="amount",
            fuel_grade=pl.when(pl.col("category") == "FUEL")
            .then(pl.col("NONSCAN_SUBCATEGORY"))
            .otherwise(None),
        )
    )
    if store_limit:
        keep = (
            lf.select("store_id").unique().head(store_limit).collect(engine="streaming")
        )
        lf = lf.join(keep.lazy(), on="store_id", how="inner")
    return lf.collect(engine="streaming")


def build_visit(lines: pl.DataFrame) -> pl.DataFrame:
    """Anchor on each fuel transaction and sweep the window around it.

    The window has no equality key, so a plain range join degenerates into a
    nested loop over tens of millions of rows. Bucketing both sides by hour
    first gives the join something to hash on; the window spans at most three
    buckets, so each candidate is offered to its own bucket and its two
    neighbours and the exact minutes are applied afterwards as a filter.
    """
    fuel = (
        lines.filter(pl.col("category") == "FUEL")
        .group_by("tx")
        .agg(
            shopper_id=pl.col("shopper_id").first(),
            store_id=pl.col("store_id").first(),
            ts=pl.col("ts").min(),
            fuel_spend=pl.col("amount").sum(),
            fuel_grade=pl.col("fuel_grade").drop_nulls().first(),
        )
        .rename({"tx": "fuel_tx"})
    )
    store = (
        lines.filter(~pl.col("category").is_in(C.NOT_A_PURCHASE))
        .group_by("tx")
        .agg(
            shopper_id=pl.col("shopper_id").first(),
            ts=pl.col("ts").min(),
            cats=pl.col("category").unique(),
            spend=pl.col("amount").sum(),
        )
        .rename({"tx": "store_tx"})
    )

    fuel_b = fuel.with_columns(bucket=pl.col("ts").dt.truncate("1h"))
    store_b = store.with_columns(bucket=pl.col("ts").dt.truncate("1h"))
    shifted = pl.concat(
        [
            store_b.with_columns(bucket=pl.col("bucket") + pl.duration(hours=k))
            for k in (-1, 0, 1)
        ]
    )

    delta = (pl.col("ts_store") - pl.col("ts")).dt.total_minutes()
    hit = (
        fuel_b.join(
            shifted.select(
                "shopper_id", "bucket", "store_tx", "cats", "spend", ts_store="ts"
            ),
            on=["shopper_id", "bucket"],
            how="inner",
        )
        .filter((delta >= -C.WINDOW_BEFORE) & (delta <= C.WINDOW_AFTER))
    )

    rolled = hit.group_by("fuel_tx").agg(
        cats=pl.col("cats")
        .list.explode(keep_nulls=False, empty_as_null=False)
        .unique(),
        store_spend=pl.col("spend").sum(),
        n_store_tx=pl.col("store_tx").n_unique(),
    )
    return (
        fuel.join(rolled, on="fuel_tx", how="left")
        .with_columns(
            bought=pl.col("n_store_tx").fill_null(0) > 0,
            store_spend=pl.col("store_spend").fill_null(0.0),
            hour=pl.col("ts").dt.hour(),
            weekday=pl.col("ts").dt.weekday(),
            month=pl.col("ts").dt.month(),
            daypart=daypart_expr(pl.col("ts")),
        )
        .drop("n_store_tx")
        .sort("ts")
    )


def feat_history(visit: pl.DataFrame) -> pl.DataFrame:
    """One row per visit x target category, using only prior visits."""
    frames = []
    for cat in C.TARGET_CATEGORIES:
        hit = pl.col("cats").list.contains(cat).fill_null(False).cast(pl.Int8)
        over = pl.col("shopper_id")
        df = (
            visit.select("fuel_tx", "shopper_id", "ts", "cats")
            .with_columns(hit=hit)
            .drop("cats")
            # fuel_tx breaks ties: without it, two visits sharing a
            # timestamp order arbitrarily and the "prior" window is not
            # reproducible between runs.
            .sort("shopper_id", "ts", "fuel_tx")
            .with_columns(
                prior_n=pl.col("hit").cum_count().over(over).cast(pl.Int32) - 1,
                prior_hits=(pl.col("hit").cum_sum().over(over) - pl.col("hit")),
            )
            .with_columns(
                bought_before=(pl.col("prior_hits") > 0).cast(pl.Int8),
                prior_share=pl.when(pl.col("prior_n") > 0)
                .then(pl.col("prior_hits") / pl.col("prior_n"))
                .otherwise(0.0),
                category=pl.lit(cat),
            )
            .select("fuel_tx", "category", "prior_n", "bought_before", "prior_share")
        )
        frames.append(df)
    return pl.concat(frames)


def build_als_interactions(lines: pl.DataFrame) -> pl.DataFrame:
    """shopper x (sku @ daypart) purchase counts, for the SKU model.

    Restricted to the training period. ALS feels like preprocessing rather
    than a model, which is exactly why this gets forgotten -- factor the
    whole two years and the latent factors have already seen the purchases
    they are being asked to predict.
    """
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    return (
        lines.filter(
            pl.col("category").is_in(C.SKU_MODEL_CATEGORIES)
            & pl.col("gtin").is_not_null()
            & (pl.col("ts") < cut)
        )
        .with_columns(
            item=pl.concat_str(
                [pl.col("gtin"), daypart_expr(pl.col("ts"))], separator="@"
            )
        )
        .group_by("shopper_id", "item")
        .agg(n=pl.len().cast(pl.Float64))
    )


def build_offer_catalog(lines: pl.DataFrame) -> pl.DataFrame:
    """Candidate offers, one per SKU, with ASSUMED economics.

    The real version of this table comes from the client: unit margin, the
    funded discount, and whether a vendor is paying for it. None of that
    exists in the PDI extract, so the numbers here are a placeholder that
    makes the decision rule runnable -- not a finding.
    """
    return (
        lines.filter(
            pl.col("category").is_in(C.SKU_MODEL_CATEGORIES)
            & pl.col("gtin").is_not_null()
        )
        .group_by("gtin", "category")
        .agg(
            brand=pl.col("brand").drop_nulls().first(),
            avg_price=pl.col("amount").mean().round(2),
            units=pl.len(),
        )
        .filter(pl.col("units") >= 500)
        .with_columns(
            unit_margin=(pl.col("avg_price") * C.ASSUMED_MARGIN_RATE).round(3),
            discount=pl.lit(0.90),
            # How much of the discount the BRAND pays. This is a share, not
            # a flag: a brand often funds all of it, sometimes the retailer
            # and the brand split it. At a full 100% the store's margin per
            # unit is unchanged and it simply sells more volume, which is
            # why these programmes work at all.
            funded_share=pl.when(
                pl.col("brand").is_in(["Monster", "Red Bull", "Rockstar"])
            ).then(pl.lit(1.00))        # national energy-drink programmes
            .when(pl.col("brand").is_in(["Gatorade", "Coca-Cola", "Pepsi"]))
            .then(pl.lit(0.50))         # typical split with the retailer
            .otherwise(pl.lit(0.00)),   # nobody is funding it but us
            in_stock=pl.lit(True),
        )
        .with_columns(
            our_cost=((1 - pl.col("funded_share")) * pl.col("discount")).round(3)
        )
        .sort("units", descending=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", type=int, default=None,
                    help="build from a subset of stores, for a fast dry run")
    args = ap.parse_args()

    t = log("reading bronze and cleaning lines")
    lines = build_line_clean(args.stores)
    t = log(f"line_clean: {lines.height:,} rows", t)
    lines.write_parquet(C.ART / "line_clean.parquet")

    visit = build_visit(lines)
    rate = visit["bought"].mean()
    t = log(f"visit: {visit.height:,} fuel visits, attach {rate:.1%}", t)
    visit.write_parquet(C.ART / "visit.parquet")

    hist = feat_history(visit)
    t = log(f"feat_history: {hist.height:,} rows", t)
    hist.write_parquet(C.ART / "feat_history.parquet")

    als = build_als_interactions(lines)
    t = log(f"als_interactions: {als.height:,} shopper-item pairs", t)
    als.write_parquet(C.ART / "als_interactions.parquet")

    cat = build_offer_catalog(lines)
    t = log(f"offer_catalog: {cat.height:,} candidate SKUs (margins ASSUMED)", t)
    cat.write_csv(C.ART / "offer_catalog.csv")

    log("done. next: train_models.py")


if __name__ == "__main__":
    main()
