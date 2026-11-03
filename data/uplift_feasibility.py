"""Can we estimate uplift from the discounts table?

The blocker: the table records REDEMPTIONS, not impressions. A row exists
only when a discount was applied to a purchase, so "got a discount" implies
"bought". There is no denominator -- no record of an offer shown and
declined -- and therefore no P(buy | offered).

The workaround, if the data supports it: promotions turn on and off by store
and by week. That natural variation gives a difference-in-differences design.
This script checks whether that variation actually exists.
"""
import polars as pl

D = "/Users/hathawayj/git/byuibigdata/bdbook/data"

dis = (pl.scan_parquet(f"{D}/discounts/*.parquet")
       .with_columns(
           day=pl.col("DATE_TIME").dt.date(),
           store=pl.col("STORE_ID").cast(pl.Int64),
           dtype=pl.col("DISCOUNT_TYPE").cast(pl.Int64),
           amt=pl.col("DISCOUNT_AMOUNT").cast(pl.Float64).abs()))

print("=== 1. Discount depth by type (the 'dose')")
print(dis.group_by("dtype").agg(
        uses=pl.len(),
        median_amt=pl.col("amt").median().round(2),
        mean_amt=pl.col("amt").mean().round(2),
        stores=pl.col("store").n_unique(),
     ).sort("uses", descending=True).collect(engine="streaming"))

print("\n=== 2. Do individual promotions switch on and off by store?")
promo = (dis.filter(pl.col("DISCOUNT_NAME").is_not_null())
    .group_by("DISCOUNT_NAME").agg(
        uses=pl.len(),
        stores=pl.col("store").n_unique(),
        first=pl.col("day").min(),
        last=pl.col("day").max(),
        days_active=pl.col("day").n_unique(),
        median_amt=pl.col("amt").median().round(2))
    .filter(pl.col("uses") > 20000)
    .sort("uses", descending=True).head(18))
with pl.Config(tbl_rows=20, fmt_str_lengths=32, tbl_width_chars=140):
    print(promo.collect(engine="streaming"))

print("\n=== 3. Staggered rollout? First day each store used a big promo")
for name in ["Skupos", "loyaltyDiscount", "MONSTER"]:
    starts = (dis.filter(pl.col("DISCOUNT_NAME") == name)
        .group_by("store").agg(first=pl.col("day").min(), uses=pl.len())
        .filter(pl.col("uses") > 50)
        .collect(engine="streaming"))
    if starts.height == 0:
        print(f"  {name}: too few stores")
        continue
    q = starts["first"]
    print(f"  {name:<18} {starts.height:>4} stores   "
          f"first starts {q.min()}  median {q.median()}  last {q.max()}")

print("\n=== 4. The denominator problem, stated numerically")
tx_with_disc = dis.select(pl.col("TRANSACTION_SET_ID").n_unique()).collect().item()
print(f"  transactions with a discount row: {tx_with_disc:,}")
print("  transactions where an offer was SHOWN but not taken:  0")
print("  -> P(buy | offered) is not computable from this table.")
