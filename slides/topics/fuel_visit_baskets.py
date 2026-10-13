"""Do fuel and in-store purchases ring up as separate transactions?

A basket keyed on TRANSACTION_SET_ID can only see what was rung up together. At a
C-Store the pump and the counter are often two transactions by the same person a
few minutes apart, so "fuel + drink" never appears in a transaction-level basket.

This script stitches transactions into *visits* (same shopper, same store, within
a time window) and compares the two grains.

Run:  uv run --with polars python fuel_visit_baskets.py [path-to-dewey_data]
"""

import sys

import polars as pl

DATA = sys.argv[1] if len(sys.argv) > 1 else "/Users/hathawayj/git/byuibigdata/dewey_data"
WINDOWS = [5, 15, 30, 60]  # minutes between transactions that still count as one visit
GAP = 30  # the window used for the headline numbers


def category_lines() -> pl.DataFrame:
    """Clean line items, labelled with a category, one row per line."""
    items = pl.read_parquet(f"{DATA}/pdi_cstore_transection_items/*.parquet")
    gtin = pl.read_parquet(f"{DATA}/pdi_one/cstore_master_ctin.parquet").select("GTIN", "CATEGORY")
    return (
        items.filter((pl.col("UNIT_QUANTITY") > 0) & (pl.col("GRAND_TOTAL_AMOUNT") > 0))
        .join(gtin, on="GTIN", how="left")
        .with_columns(category=pl.coalesce("CATEGORY", "NONSCAN_CATEGORY"))
        .drop_nulls("category")
        .select("TRANSACTION_SET_ID", "DATE_TIME", "category", "GRAND_TOTAL_AMOUNT")
    )


def with_shopper(lines: pl.DataFrame) -> pl.DataFrame:
    """Attach shopper and store. Only transactions with a shopper id survive."""
    shopper = (
        pl.read_parquet(f"{DATA}/pdi_cstore/cstore_shopper.parquet")
        .select("TRANSACTION_SET_ID", "SHOPPER_ID")
        .unique(subset="TRANSACTION_SET_ID")
    )
    store = (
        pl.read_parquet(f"{DATA}/pdi_cstore/cstore_payments.parquet")
        .select("TRANSACTION_SET_ID", "STORE_ID")
        .unique(subset="TRANSACTION_SET_ID")
    )
    return lines.join(shopper, on="TRANSACTION_SET_ID").join(store, on="TRANSACTION_SET_ID")


def visits(lines: pl.DataFrame, gap_minutes: int) -> pl.DataFrame:
    """Group transactions into visits: same shopper, same store, gap under the window."""
    tx = (
        lines.group_by("SHOPPER_ID", "STORE_ID", "TRANSACTION_SET_ID")
        .agg(pl.col("DATE_TIME").min().alias("ts"))
        .sort("SHOPPER_ID", "STORE_ID", "ts")
    )
    tx = tx.with_columns(
        new_visit=(
            (pl.col("ts") - pl.col("ts").shift(1)).over(["SHOPPER_ID", "STORE_ID"])
            > pl.duration(minutes=gap_minutes)
        )
        .fill_null(True)
        .cast(pl.Int32)
    ).with_columns(visit_id=pl.col("new_visit").cum_sum().over(["SHOPPER_ID", "STORE_ID"]))
    return tx.select("TRANSACTION_SET_ID", "SHOPPER_ID", "STORE_ID", "visit_id")


def basket_stats(baskets: pl.DataFrame, label: str) -> dict:
    """Support, confidence and lift for FUEL -> any in-store category."""
    n = baskets.height
    has_fuel = pl.col("items").list.contains("FUEL")
    in_store = (
        pl.col("items").list.eval(pl.element().filter(pl.element() != "FUEL")).list.len() > 0
    )
    both = baskets.filter(has_fuel & in_store).height
    fuel = baskets.filter(has_fuel).height
    store_any = baskets.filter(in_store).height
    confidence = both / fuel if fuel else 0
    base = store_any / n if n else 0
    return {
        "grain": label,
        "baskets": n,
        "with_fuel": fuel,
        "fuel_and_instore": both,
        "attach_rate": round(confidence, 3),
        "base_rate": round(base, 3),
        "lift": round(confidence / base, 2) if base else 0,
    }


def fuel_pairs(baskets: pl.DataFrame, label: str, min_support: float = 0.002) -> pl.DataFrame:
    """Lift of FUEL -> each individual category."""
    n = baskets.height
    exploded = baskets.select("items").explode("items").rename({"items": "category"})
    support = (
        exploded.group_by("category")
        .agg((pl.len() / n).alias("support"))
        .filter(pl.col("category") != "FUEL")
    )
    fuel_baskets = baskets.filter(pl.col("items").list.contains("FUEL"))
    n_fuel = fuel_baskets.height
    paired = (
        fuel_baskets.select("items")
        .explode("items")
        .rename({"items": "category"})
        .filter(pl.col("category") != "FUEL")
        .group_by("category")
        .agg((pl.len() / n_fuel).alias("confidence"))
    )
    return (
        paired.join(support, on="category")
        .with_columns(lift=(pl.col("confidence") / pl.col("support")).round(2))
        .with_columns(grain=pl.lit(label))
        .filter(pl.col("support") > min_support)
        .sort("lift", descending=True)
    )


def main() -> None:
    lines = category_lines()
    print(f"clean category lines: {lines.height:,}")

    linked = with_shopper(lines)
    print(
        f"lines with a shopper id: {linked.height:,} "
        f"({linked.height / lines.height:.1%} of clean lines)"
    )
    print(f"shoppers: {linked['SHOPPER_ID'].n_unique():,}  stores: {linked['STORE_ID'].n_unique():,}")

    # --- transaction grain (what the slides used)
    tx_baskets = linked.group_by("TRANSACTION_SET_ID").agg(
        pl.col("category").unique().alias("items")
    )
    print("\n--- transaction grain")
    print(basket_stats(tx_baskets, "transaction"))

    # --- visit grain, across several windows
    print("\n--- visit grain, by stitching window")
    rows = []
    for minutes in WINDOWS:
        v = visits(linked, minutes)
        joined = linked.join(v.drop("SHOPPER_ID", "STORE_ID"), on="TRANSACTION_SET_ID")
        visit_baskets = joined.group_by("SHOPPER_ID", "STORE_ID", "visit_id").agg(
            pl.col("category").unique().alias("items"),
            pl.col("TRANSACTION_SET_ID").n_unique().alias("n_tx"),
        )
        stats = basket_stats(visit_baskets, f"visit ({minutes} min)")
        stats["multi_tx_visits"] = visit_baskets.filter(pl.col("n_tx") > 1).height
        stats["pct_multi_tx"] = round(
            visit_baskets.filter(pl.col("n_tx") > 1).height / visit_baskets.height, 4
        )
        rows.append(stats)
        print(stats)

    # --- which categories ride along with fuel, at each grain
    v = visits(linked, GAP)
    joined = linked.join(v.drop("SHOPPER_ID", "STORE_ID"), on="TRANSACTION_SET_ID")
    visit_baskets = joined.group_by("SHOPPER_ID", "STORE_ID", "visit_id").agg(
        pl.col("category").unique().alias("items")
    )
    print(f"\n--- FUEL -> category, transaction grain")
    print(fuel_pairs(tx_baskets, "transaction").head(10))
    print(f"\n--- FUEL -> category, visit grain ({GAP} min)")
    print(fuel_pairs(visit_baskets, f"visit {GAP}min").head(10))

    # --- how much stitching actually happened
    sizes = (
        joined.group_by("SHOPPER_ID", "STORE_ID", "visit_id")
        .agg(pl.col("TRANSACTION_SET_ID").n_unique().alias("n_tx"))
    )
    print("\n--- transactions per visit (%d min)" % GAP)
    print(sizes["n_tx"].value_counts(sort=True).head(6))


if __name__ == "__main__":
    main()
