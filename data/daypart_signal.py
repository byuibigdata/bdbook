"""Does SKU choice inside Packaged Beverages actually move with daypart?

If it does, encoding time into the ALS key is worth the sparsity cost.
If it doesn't, time belongs in Model B and the re-ranker, not in ALS.

Test: for shoppers who buy the category more than once, how often do they
repeat the SAME SKU when the two purchases fall in the same daypart, versus
in different dayparts? If time matters, same-daypart pairs should repeat
noticeably more often.
"""
import polars as pl

D = "/Users/hathawayj/git/byuibigdata/bdbook/data"
CAT = "Packaged Beverages (Non-Alcoholic)"

g = (pl.scan_parquet(f"{D}/master-gtin/*.parquet")
       .select("GTIN", "CATEGORY", "BRAND").unique(subset="GTIN"))
sh = (pl.scan_parquet(f"{D}/shopper-id/*.parquet")
        .select("TRANSACTION_SET_ID", "SHOPPER_ID").unique(subset="TRANSACTION_SET_ID"))
lines = (pl.scan_parquet(f"{D}/transaction-items-all/*.parquet")
    .filter((pl.col("UNIT_QUANTITY").cast(pl.Float64) > 0)
            & (pl.col("GRAND_TOTAL_AMOUNT").cast(pl.Float64) > 0)
            & (pl.col("SCAN_TYPE") == "GTIN"))
    .select("TRANSACTION_SET_ID", "GTIN", "DATE_TIME")
    .join(g, on="GTIN", how="inner").filter(pl.col("CATEGORY") == CAT)
    .join(sh, on="TRANSACTION_SET_ID")
    .with_columns(
        # Three dayparts, wrapping overnight: attach jumps at 5am and falls
        # after 11pm, and that tail is 3.2% of visits.
        daypart=pl.when(pl.col("DATE_TIME").dt.hour().is_between(5, 9))
        .then(pl.lit("1 morning"))
        .when(pl.col("DATE_TIME").dt.hour().is_between(10, 14))
        .then(pl.lit("2 midday"))
        .otherwise(pl.lit("3 evening"))
    ).select("SHOPPER_ID", "GTIN", "BRAND", "daypart"))

df = lines.collect(engine="streaming")
print(f"beverage lines: {df.height:,}")

print("\n=== 1. Does the CATEGORY-WIDE mix shift by daypart?")
top = df.group_by("BRAND").agg(n=pl.len()).sort("n", descending=True).head(8)["BRAND"]
mix = (df.filter(pl.col("BRAND").is_in(top))
         .group_by("daypart", "BRAND").agg(n=pl.len())
         .with_columns(share=pl.col("n") / pl.col("n").sum().over("daypart"))
         .pivot(on="daypart", index="BRAND", values="share")
         .sort("BRAND"))
with pl.Config(tbl_rows=12, fmt_float="full", float_precision=3):
    print(mix)

print("\n=== 2. Does an INDIVIDUAL's choice shift by daypart?")
# Keep shoppers with enough history to form pairs.
act = (df.group_by("SHOPPER_ID").agg(n=pl.len())
         .filter(pl.col("n").is_between(2, 60)))
sub = df.join(act.select("SHOPPER_ID"), on="SHOPPER_ID")
print(f"shoppers with 2-60 beverage purchases: {act.height:,}")

# Self-join each shopper's purchases into pairs, then compare.
pairs = (sub.with_row_index("i")
    .join(sub.with_row_index("j"), on="SHOPPER_ID", suffix="_b")
    .filter(pl.col("i") < pl.col("j"))
    .select(
        same_sku=pl.col("GTIN") == pl.col("GTIN_b"),
        same_brand=pl.col("BRAND") == pl.col("BRAND_b"),
        same_daypart=pl.col("daypart") == pl.col("daypart_b"),
    ))
out = (pairs.group_by("same_daypart")
    .agg(pairs=pl.len(),
         repeat_sku=pl.col("same_sku").mean(),
         repeat_brand=pl.col("same_brand").mean())
    .sort("same_daypart", descending=True))
with pl.Config(fmt_float="full", float_precision=4):
    print(out)

r = {row["same_daypart"]: row for row in out.to_dicts()}
if True in r and False in r:
    ds = r[True]["repeat_sku"] - r[False]["repeat_sku"]
    db = r[True]["repeat_brand"] - r[False]["repeat_brand"]
    print(f"\n  same-daypart advantage:  SKU {ds:+.4f}  "
          f"({ds / r[False]['repeat_sku']:+.1%} relative)")
    print(f"                           brand {db:+.4f} "
          f"({db / r[False]['repeat_brand']:+.1%} relative)")
