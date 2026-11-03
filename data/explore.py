"""Explore the PDI C-Store parquet data with Polars.

Four datasets live beside this script, about 8 GB on disk and 290M rows:

    transaction-items-all/  line items, one row per item rung up
    payments/               one row per tender on a transaction
    shopper-id/             payment -> shopper, the loyalty/card link
    discounts/              one row per discount applied to a line item
    master-gtin/            product lookup: GTIN -> category, brand, size
    stores-information/     one row per store: chain, address, lat/long
    stores-status/          the same stores, plus active status flags

Everything here stays lazy and collects with the streaming engine, so a section
runs on a laptop without loading its dataset into memory.

Run one section, or all of them:

    uv run --with polars python explore.py            # everything
    uv run --with polars python explore.py schema     # just the schemas
    uv run --with polars python explore.py shoppers baskets

Sections: schema, time, items, products, stores, payments, shoppers,
          baskets, discounts
"""

import sys
from pathlib import Path

import polars as pl

DATA = Path(__file__).parent
SETS = {
    "items": "transaction-items-all",
    "payments": "payments",
    "shopper": "shopper-id",
    "discounts": "discounts",
    "gtin": "master-gtin",
    "stores": "stores-information",
    "store_status": "stores-status",
}


def product_lookup() -> pl.LazyFrame:
    """GTIN -> category, one row per GTIN. Join this onto the line items."""
    return (
        scan("gtin")
        .select("GTIN", "CATEGORY", "SUBCATEGORY", "BRAND", "PRODUCT_TYPE")
        .unique(subset="GTIN")
    )


def labelled_lines() -> pl.LazyFrame:
    """Clean line items with a usable category on every row we can label.

    Scanned items get their category from master-gtin; keyed departments
    (fuel, foodservice, lottery) only ever had NONSCAN_CATEGORY. Coalescing
    the two is what gives a basket its items.
    """
    return (
        scan("items")
        .filter((pl.col("UNIT_QUANTITY") > 0) & (pl.col("GRAND_TOTAL_AMOUNT") > 0))
        .join(product_lookup(), on="GTIN", how="left")
        .with_columns(category=pl.coalesce("CATEGORY", "NONSCAN_CATEGORY"))
    )


def scan(name: str) -> pl.LazyFrame:
    """Lazy scan of one dataset. Decimals become floats so arithmetic behaves."""
    lf = pl.scan_parquet(DATA / SETS[name] / "*.parquet")
    decimals = [
        c for c, t in lf.collect_schema().items() if isinstance(t, pl.Decimal)
    ]
    return lf.with_columns(pl.col(c).cast(pl.Float64) for c in decimals)


def show(lf: pl.LazyFrame, title: str) -> pl.DataFrame:
    """Collect with the streaming engine and print under a heading."""
    df = lf.collect(engine="streaming")
    print(f"\n--- {title}")
    with pl.Config(tbl_rows=25, tbl_width_chars=120, fmt_str_lengths=40):
        print(df)
    return df


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def section_schema() -> None:
    """What columns exist, what types, and how many rows."""
    print("=== schemas and row counts")
    for key, folder in SETS.items():
        lf = pl.scan_parquet(DATA / folder / "*.parquet")
        schema = lf.collect_schema()
        rows = lf.select(pl.len()).collect().item()
        print(f"\n{folder}  —  {rows:,} rows, {len(schema)} columns")
        for col, dtype in schema.items():
            print(f"    {col:<24} {dtype}")


def section_time() -> None:
    """How far back the data goes, and how it is distributed by month."""
    print("\n=== time span")
    show(
        scan("items").select(
            start=pl.col("DATE_TIME").min(),
            end=pl.col("DATE_TIME").max(),
            days=(pl.col("DATE_TIME").max() - pl.col("DATE_TIME").min()).dt.total_days(),
        ),
        "line item date range",
    )
    show(
        scan("items")
        .group_by(month=pl.col("DATE_TIME").dt.truncate("1mo"))
        .agg(
            lines=pl.len(),
            transactions=pl.col("TRANSACTION_SET_ID").n_unique(),
            stores=pl.col("STORE_ID").n_unique(),
        )
        .sort("month"),
        "volume by month — watch for partial months at either end",
    )


def section_items() -> None:
    """The line item table: how items are identified and categorised."""
    print("\n=== line items")
    show(
        scan("items").select(
            lines=pl.len(),
            transactions=pl.col("TRANSACTION_SET_ID").n_unique(),
            stores=pl.col("STORE_ID").n_unique(),
            gtins=pl.col("GTIN").n_unique(),
            null_gtin=pl.col("GTIN").null_count(),
            null_category=pl.col("NONSCAN_CATEGORY").null_count(),
        ),
        "shape and identifier coverage",
    )
    show(
        scan("items")
        .group_by("SCAN_TYPE")
        .agg(lines=pl.len(), revenue=pl.col("GRAND_TOTAL_AMOUNT").sum())
        .sort("lines", descending=True),
        "SCAN_TYPE — scanned barcodes vs. keyed departments",
    )
    show(
        scan("items")
        .filter(pl.col("NONSCAN_CATEGORY").is_not_null())
        .group_by("NONSCAN_CATEGORY")
        .agg(
            lines=pl.len(),
            revenue=pl.col("GRAND_TOTAL_AMOUNT").sum().round(0),
            avg_line=pl.col("GRAND_TOTAL_AMOUNT").mean().round(2),
        )
        .sort("revenue", descending=True)
        .head(20),
        "top NONSCAN_CATEGORY by revenue — this is where FUEL shows up",
    )
    show(
        scan("items")
        .group_by("POS_DESCRIPTION")
        .agg(lines=pl.len())
        .sort("lines", descending=True)
        .head(20),
        "most common POS_DESCRIPTION",
    )
    show(
        scan("items").select(
            negative_qty=(pl.col("UNIT_QUANTITY") <= 0).sum(),
            negative_amount=(pl.col("GRAND_TOTAL_AMOUNT") <= 0).sum(),
            max_amount=pl.col("GRAND_TOTAL_AMOUNT").max(),
        ),
        "returns and voids — filter these before any basket work",
    )


def section_payments() -> None:
    """How people pay, and how many tenders a transaction carries."""
    print("\n=== payments")
    show(
        scan("payments")
        .group_by("PAYMENT_TYPE")
        .agg(payments=pl.len(), tender=pl.col("TENDER").sum().round(0))
        .sort("payments", descending=True)
        .head(15),
        "PAYMENT_TYPE",
    )
    show(
        scan("payments")
        .group_by("CARD_TYPE")
        .agg(payments=pl.len())
        .sort("payments", descending=True)
        .head(15),
        "CARD_TYPE — the loyalty link rides on these",
    )
    show(
        scan("payments")
        .group_by("TRANSACTION_SET_ID")
        .agg(n=pl.len())
        .group_by("n")
        .agg(transactions=pl.len())
        .sort("n")
        .head(8),
        "tenders per transaction — split tender breaks a naive join",
    )


def section_shoppers() -> None:
    """Who we can identify, and how much history each shopper has."""
    print("\n=== shoppers")
    linked = (
        scan("shopper")
        .select("TRANSACTION_SET_ID", "SHOPPER_ID")
        .unique(subset="TRANSACTION_SET_ID")
    )
    show(
        linked.select(
            linked_transactions=pl.col("TRANSACTION_SET_ID").n_unique(),
            shoppers=pl.col("SHOPPER_ID").n_unique(),
        ),
        "identified transactions and distinct shoppers",
    )

    per_shopper = (
        scan("items")
        .select("TRANSACTION_SET_ID", "DATE_TIME")
        .group_by("TRANSACTION_SET_ID")
        .agg(ts=pl.col("DATE_TIME").min())
        .join(linked, on="TRANSACTION_SET_ID", how="inner")
        .group_by("SHOPPER_ID")
        .agg(
            visits=pl.len(),
            first=pl.col("ts").min(),
            last=pl.col("ts").max(),
        )
        .with_columns(
            tenure_days=(pl.col("last") - pl.col("first")).dt.total_days()
        )
        .collect(engine="streaming")
    )

    print(f"\n--- history per shopper  ({per_shopper.height:,} shoppers)")
    print(per_shopper.select("visits", "tenure_days").describe())

    n = per_shopper.height
    print("\n    how many shoppers come back?")
    for k in (2, 5, 10, 25):
        c = per_shopper.filter(pl.col("visits") >= k).height
        print(f"      >= {k:>2} visits:  {c:>10,}  ({c / n:6.1%})")
    print("\n    how long do we observe them?")
    for d in (30, 180, 365):
        c = per_shopper.filter(pl.col("tenure_days") >= d).height
        print(f"      >= {d:>3} days:   {c:>10,}  ({c / n:6.1%})")
    print(
        "\n    A two-year dataset is not two years per customer. The median "
        "shopper\n    is the constraint on any personalised model."
    )


def section_baskets() -> None:
    """Basket size, value, and how often fuel rings up with anything else."""
    print("\n=== baskets")
    clean = scan("items").filter(
        (pl.col("UNIT_QUANTITY") > 0) & (pl.col("GRAND_TOTAL_AMOUNT") > 0)
    )
    baskets = clean.group_by("TRANSACTION_SET_ID").agg(
        lines=pl.len(),
        categories=pl.col("NONSCAN_CATEGORY").n_unique(),
        total=pl.col("GRAND_TOTAL_AMOUNT").sum(),
        has_fuel=pl.col("NONSCAN_CATEGORY").str.contains("(?i)fuel").any(),
    )
    show(
        baskets.select(
            baskets=pl.len(),
            single_line=(pl.col("lines") == 1).mean().round(3),
            median_lines=pl.col("lines").median(),
            median_total=pl.col("total").median().round(2),
            mean_total=pl.col("total").mean().round(2),
        ),
        "basket shape — single-line baskets carry no association signal",
    )
    show(
        baskets.group_by("has_fuel")
        .agg(
            baskets=pl.len(),
            median_total=pl.col("total").median().round(2),
            median_lines=pl.col("lines").median(),
        )
        .sort("has_fuel"),
        "fuel vs. non-fuel baskets, at the TRANSACTION grain",
    )
    print(
        "\n    Note: pump and counter are often two transactions minutes apart.\n"
        "    Stitch them into visits by shopper and store before concluding\n"
        "    anything about fuel attach. See slides/topics/fuel_visit_baskets.py."
    )


def section_discounts() -> None:
    """What promotions already run, and what they cost."""
    print("\n=== discounts")
    show(
        scan("discounts").select(
            rows=pl.len(),
            transactions=pl.col("TRANSACTION_SET_ID").n_unique(),
            total_discount=pl.col("DISCOUNT_AMOUNT").sum().round(0),
        ),
        "shape",
    )
    show(
        scan("discounts")
        .group_by("DISCOUNT_NAME")
        .agg(
            times_used=pl.len(),
            total=pl.col("DISCOUNT_AMOUNT").sum().round(0),
            avg=pl.col("DISCOUNT_AMOUNT").mean().round(2),
        )
        .sort("times_used", descending=True)
        .head(20),
        "most-used discounts — the offers the business already funds",
    )


def section_products() -> None:
    """The GTIN master: what it covers, and how much of the till it labels."""
    print("\n=== products (master-gtin)")
    show(
        scan("gtin").select(
            rows=pl.len(),
            gtins=pl.col("GTIN").n_unique(),
            categories=pl.col("CATEGORY").n_unique(),
            subcategories=pl.col("SUBCATEGORY").n_unique(),
            brands=pl.col("BRAND").n_unique(),
        ),
        "shape",
    )
    show(
        scan("gtin")
        .group_by("CATEGORY")
        .agg(products=pl.len())
        .sort("products", descending=True)
        .head(20),
        "products per CATEGORY",
    )

    # The number that decides whether basket work is viable.
    matched = (
        scan("items")
        .join(product_lookup(), on="GTIN", how="left")
        .select(
            lines=pl.len(),
            gtin_lines=(pl.col("SCAN_TYPE") == "GTIN").sum(),
            matched_master=pl.col("CATEGORY").is_not_null().sum(),
            has_nonscan=pl.col("NONSCAN_CATEGORY").is_not_null().sum(),
            labelled_either=pl.coalesce("CATEGORY", "NONSCAN_CATEGORY")
            .is_not_null()
            .sum(),
        )
    )
    df = show(matched, "how many line items end up with a category?")
    n = df["lines"][0]
    for col in ("gtin_lines", "matched_master", "has_nonscan", "labelled_either"):
        print(f"    {col:<18} {df[col][0]:>12,}  ({df[col][0] / n:6.1%} of lines)")
    print(
        "\n    Unmatched GTINs are mostly local or store-specific items. Decide\n"
        "    early whether they become an 'UNKNOWN' basket item or get dropped."
    )

    show(
        labelled_lines()
        .group_by("category")
        .agg(
            lines=pl.len(),
            revenue=pl.col("GRAND_TOTAL_AMOUNT").sum().round(0),
            avg_line=pl.col("GRAND_TOTAL_AMOUNT").mean().round(2),
        )
        .sort("revenue", descending=True)
        .head(25),
        "top categories by revenue, after coalescing master + nonscan",
    )


def section_stores() -> None:
    """The store master: geography, chains, and which stores actually transact."""
    print("\n=== stores")
    show(
        scan("stores").select(
            rows=pl.len(),
            stores=pl.col("STORE_ID").n_unique(),
            chains=pl.col("STORE_CHAIN_ID").n_unique(),
            states=pl.col("STATE").n_unique(),
            null_latlong=pl.col("LATITUDE").null_count(),
        ),
        "shape",
    )
    show(
        scan("stores")
        .group_by("STATE")
        .agg(stores=pl.len())
        .sort("stores", descending=True)
        .head(15),
        "stores per state",
    )
    show(
        scan("stores")
        .group_by("STORE_CHAIN_NAME")
        .agg(stores=pl.len(), chain_size=pl.col("CHAIN_SIZE").max())
        .sort("stores", descending=True)
        .head(15),
        "largest chains in the master",
    )
    show(
        scan("store_status")
        .group_by("ACTIVE_STATUS", "STORE_FLAG")
        .agg(stores=pl.len())
        .sort("stores", descending=True)
        .head(10),
        "stores-status — the extra columns over stores-information",
    )

    # The master covers far more stores than the transaction data does.
    transacting = scan("items").select(pl.col("STORE_ID").unique())
    show(
        transacting.join(
            scan("stores").select("STORE_ID", "STATE", "STORE_CHAIN_NAME").unique(
                subset="STORE_ID"
            ),
            on="STORE_ID",
            how="left",
        ).select(
            transacting_stores=pl.len(),
            found_in_master=pl.col("STATE").is_not_null().sum(),
        ),
        "do the stores in our transactions appear in the master?",
    )
    show(
        transacting.join(
            scan("stores").select("STORE_ID", "STATE").unique(subset="STORE_ID"),
            on="STORE_ID",
            how="inner",
        )
        .group_by("STATE")
        .agg(stores=pl.len())
        .sort("stores", descending=True)
        .head(15),
        "where our transacting stores actually are",
    )



SECTIONS = {
    "schema": section_schema,
    "time": section_time,
    "items": section_items,
    "products": section_products,
    "stores": section_stores,
    "payments": section_payments,
    "shoppers": section_shoppers,
    "baskets": section_baskets,
    "discounts": section_discounts,
}


def main() -> None:
    wanted = sys.argv[1:] or list(SECTIONS)
    unknown = [w for w in wanted if w not in SECTIONS]
    if unknown:
        sys.exit(f"unknown section(s): {', '.join(unknown)}\navailable: {', '.join(SECTIONS)}")
    for name in wanted:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
