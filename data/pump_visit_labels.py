"""Build the fuel-visit table that a pump-screen model would train on.

The question is: a shopper is at the pump. Will they come inside, and what
will they buy? That label does not exist in the data -- the pump and the
counter are separate transactions -- so it has to be constructed.

A *joint basket* is anchored on the fuel transaction: every purchase by the
same shopper from 15 minutes before it to 45 minutes after. The window is
asymmetric on purpose. Fifteen minutes before catches the customer who
grabs a drink and then fills up; forty-five after catches the fill-then-shop
order, which is the common one, with room for a queue.

The label is whether that window contains a real in-store purchase, and in
which categories.

One trap this script handles: `SERVICES` is 99.5% card-processing fees of
about $0.35, charged on the fuel sale itself. Counting it as an in-store
purchase leaks the fuel transaction into its own label and overstates the
attach rate by seven points.

Run:  uv run --with polars python pump_visit_labels.py
"""

import polars as pl

DATA = "/Users/hathawayj/git/byuibigdata/bdbook/data"
BEFORE = 15  # minutes before the fuel transaction that still count
AFTER = 45   # minutes after
NOT_A_PURCHASE = ["FUEL", "SERVICES"]


def lines() -> pl.LazyFrame:
    """Clean line items with a category and a shopper, one row per line."""
    gtin = (
        pl.scan_parquet(f"{DATA}/master-gtin/*.parquet")
        .select("GTIN", "CATEGORY")
        .unique(subset="GTIN")
    )
    shopper = (
        pl.scan_parquet(f"{DATA}/shopper-id/*.parquet")
        .select("TRANSACTION_SET_ID", "SHOPPER_ID")
        .unique(subset="TRANSACTION_SET_ID")
    )
    return (
        pl.scan_parquet(f"{DATA}/transaction-items-all/*.parquet")
        .filter(
            (pl.col("UNIT_QUANTITY").cast(pl.Float64) > 0)
            & (pl.col("GRAND_TOTAL_AMOUNT").cast(pl.Float64) > 0)
        )
        .join(gtin, on="GTIN", how="left")
        .with_columns(
            category=pl.coalesce("CATEGORY", "NONSCAN_CATEGORY"),
            amount=pl.col("GRAND_TOTAL_AMOUNT").cast(pl.Float64),
        )
        .drop_nulls("category")
        .join(shopper, on="TRANSACTION_SET_ID", how="inner")
        .select(
            "SHOPPER_ID", "STORE_ID", "TRANSACTION_SET_ID", "DATE_TIME",
            "category", "amount", "NONSCAN_SUBCATEGORY",
        )
    )


def anchors_and_candidates(df: pl.DataFrame):
    """Split the lines into fuel transactions and purchasable transactions.

    Both come back at transaction grain, which keeps the window join small.
    """
    fuel = (
        df.filter(pl.col("category") == "FUEL")
        .group_by("TRANSACTION_SET_ID")
        .agg(
            SHOPPER_ID=pl.col("SHOPPER_ID").first(),
            STORE_ID=pl.col("STORE_ID").first(),
            ts=pl.col("DATE_TIME").min(),
            fuel_spend=pl.col("amount").sum(),
            grade=pl.col("NONSCAN_SUBCATEGORY").first(),
        )
        .rename({"TRANSACTION_SET_ID": "fuel_tx"})
    )
    store = (
        df.filter(~pl.col("category").is_in(NOT_A_PURCHASE))
        .group_by("TRANSACTION_SET_ID")
        .agg(
            SHOPPER_ID=pl.col("SHOPPER_ID").first(),
            STORE_ID=pl.col("STORE_ID").first(),
            ts=pl.col("DATE_TIME").min(),
            cats=pl.col("category").unique(),
            spend=pl.col("amount").sum(),
        )
        .rename({"TRANSACTION_SET_ID": "store_tx"})
    )
    return fuel, store


def window_join(fuel: pl.DataFrame, store: pl.DataFrame) -> pl.DataFrame:
    """Attach every in-store transaction inside the window to its fuel anchor.

    A plain join on SHOPPER_ID would explode for frequent shoppers, so the
    candidates are first bucketed by hour. The window spans at most three
    buckets, so each in-store transaction is offered to the bucket it falls
    in and to its two neighbours; the exact window is then applied as a
    filter.
    """
    fuel = fuel.with_columns(bucket=pl.col("ts").dt.truncate("1h"))
    store = store.with_columns(bucket=pl.col("ts").dt.truncate("1h"))

    shifted = pl.concat(
        [
            store.with_columns(bucket=pl.col("bucket") + pl.duration(hours=offset))
            for offset in (-1, 0, 1)
        ]
    )

    joined = fuel.join(
        shifted.select(
            "SHOPPER_ID", "bucket", "store_tx", "STORE_ID", "cats", "spend",
            ts_store="ts",
        ),
        on=["SHOPPER_ID", "bucket"],
        how="left",
        suffix="_store",
    )
    delta = (pl.col("ts_store") - pl.col("ts")).dt.total_minutes()
    return joined.with_columns(
        in_window=(delta >= -BEFORE) & (delta <= AFTER),
        same_store=pl.col("STORE_ID_store") == pl.col("STORE_ID"),
        minutes=delta,
    )


def main() -> None:
    df = lines().collect(engine="streaming")
    print(f"clean labelled lines with a shopper: {df.height:,}")

    fuel, store = anchors_and_candidates(df)
    print(f"fuel transactions:     {fuel.height:,}")
    print(f"in-store transactions: {store.height:,}  (FUEL and SERVICES removed)")

    hit = window_join(fuel, store).filter(pl.col("in_window").fill_null(False))

    basket = fuel.join(
        hit.group_by("fuel_tx").agg(
            cats=pl.col("cats").list.explode(keep_nulls=False,
                                             empty_as_null=False).unique(),
            store_spend=pl.col("spend").sum(),
            n_store_tx=pl.col("store_tx").n_unique(),
            other_store=(~pl.col("same_store")).any(),
        ),
        on="fuel_tx",
        how="left",
    ).with_columns(
        bought=pl.col("n_store_tx").fill_null(0) > 0,
        store_spend=pl.col("store_spend").fill_null(0.0),
    )

    n = basket.height
    attach = basket["bought"].mean()
    print(f"\n--- THE LABEL   (window: -{BEFORE} min to +{AFTER} min)")
    print(f"fuel visits:          {n:,}")
    print(f"  bought in-store:    {basket['bought'].sum():,}  ({attach:.1%})")
    print(f"  fuel only:          {(~basket['bought']).sum():,}  ({1 - attach:.1%})")

    won = basket.filter("bought")
    print(f"\n  when they do buy:   median ${won['store_spend'].median():.2f}, "
          f"mean ${won['store_spend'].mean():.2f}")
    print(f"  at another store:   {won['other_store'].sum():,} of {won.height:,}")

    print("\n--- when, relative to the pump, does the in-store sale happen?")
    print(
        hit.with_columns(
            bin=pl.when(pl.col("minutes") < 0).then(pl.lit("1 | -15..0 before"))
            .when(pl.col("minutes") < 5).then(pl.lit("2 | 0..5"))
            .when(pl.col("minutes") < 15).then(pl.lit("3 | 5..15"))
            .when(pl.col("minutes") < 30).then(pl.lit("4 | 15..30"))
            .otherwise(pl.lit("5 | 30..45"))
        )
        .group_by("bin")
        .agg(transactions=pl.len())
        .sort("bin")
    )

    print("\n--- what they buy (share of converting visits)")
    with pl.Config(tbl_rows=20, fmt_str_lengths=38):
        print(
            won.select("cats").explode("cats")
            .group_by("cats").agg(visits=pl.len())
            .with_columns(share=(pl.col("visits") / won.height).round(4))
            .sort("visits", descending=True).head(15)
        )

    print("\n--- attach rate by fuel grade")
    print(
        basket.group_by("grade")
        .agg(visits=pl.len(), attach=pl.col("bought").mean().round(4))
        .filter(pl.col("visits") > 10000)
        .sort("attach", descending=True)
    )

    print("\n--- attach rate by hour of day (store wall clock)")
    with pl.Config(tbl_rows=24):
        print(
            basket.with_columns(hour=pl.col("ts").dt.hour())
            .group_by("hour")
            .agg(visits=pl.len(), attach=pl.col("bought").mean().round(4))
            .sort("hour")
        )

    print("\n--- how much history does a fuel visit have behind it?")
    hist = basket.sort("ts").with_columns(
        prior=pl.col("ts").cum_count().over("SHOPPER_ID") - 1
    )
    for lo, hi, label in [
        (0, 0, "no prior fuel visit"),
        (1, 2, "1-2 prior"),
        (3, 9, "3-9 prior"),
        (10, 10**9, "10+ prior"),
    ]:
        grp = hist.filter(pl.col("prior").is_between(lo, hi))
        print(f"  {label:<22} {grp.height:>12,}  ({grp.height / n:6.1%})   "
              f"attach {grp['bought'].mean():.1%}")


if __name__ == "__main__":
    main()
