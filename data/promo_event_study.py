"""Event study: does a brand promotion move that brand's share?

We cannot measure uplift per customer -- no impressions. But stores adopt
promotions at different dates, so we can ask a store-level question instead:
when a store starts running the MONSTER promotion, does Monster's share of
its beverage sales rise, relative to stores that never run it?

That is a difference-in-differences design on staggered adoption.
"""
import polars as pl

D = "/Users/hathawayj/git/byuibigdata/bdbook/data"
CAT = "Packaged Beverages (Non-Alcoholic)"

g = (pl.scan_parquet(f"{D}/master-gtin/*.parquet")
     .select("GTIN", "CATEGORY", "BRAND").unique(subset="GTIN"))

bev = (pl.scan_parquet(f"{D}/transaction-items-all/*.parquet")
    .filter((pl.col("UNIT_QUANTITY").cast(pl.Float64) > 0)
            & (pl.col("GRAND_TOTAL_AMOUNT").cast(pl.Float64) > 0)
            & (pl.col("SCAN_TYPE") == "GTIN"))
    .select("GTIN", "STORE_ID", "DATE_TIME")
    .join(g, on="GTIN", how="inner")
    .filter(pl.col("CATEGORY") == CAT)
    .with_columns(
        store=pl.col("STORE_ID").cast(pl.Int64),
        week=pl.col("DATE_TIME").dt.truncate("1w"),
        is_monster=pl.col("BRAND") == "Monster")
    .group_by("store", "week")
    .agg(bev_lines=pl.len(), monster=pl.col("is_monster").sum())
    .collect(engine="streaming"))
print(f"store-weeks of beverage sales: {bev.height:,}")

adopt = (pl.scan_parquet(f"{D}/discounts/*.parquet")
    .filter(pl.col("DISCOUNT_NAME").str.to_lowercase().str.contains("monster"))
    .with_columns(store=pl.col("STORE_ID").cast(pl.Int64),
                  week=pl.col("DATE_TIME").dt.truncate("1w"))
    .group_by("store").agg(start=pl.col("week").min(), uses=pl.len())
    .filter(pl.col("uses") > 200)
    .collect(engine="streaming"))
print(f"stores running a Monster promo: {adopt.height}")

# Treated stores must have real data on both sides of their start date.
panel = bev.join(adopt, on="store", how="left")
treated = panel.filter(pl.col("start").is_not_null()).with_columns(
    evt=((pl.col("week") - pl.col("start")).dt.total_days() / 7).round(0).cast(pl.Int32))
control = panel.filter(pl.col("start").is_null())

print(f"\n  treated store-weeks: {treated.height:,}   "
      f"control store-weeks: {control.height:,}")

print("\n=== Monster share of beverage lines, by week relative to promo start")
ev = (treated.filter(pl.col("evt").is_between(-12, 12))
      .group_by("evt")
      .agg(stores=pl.col("store").n_unique(),
           share=(pl.col("monster").sum() / pl.col("bev_lines").sum()))
      .sort("evt"))
with pl.Config(tbl_rows=30):
    print(ev.with_columns(share=(pl.col("share") * 100).round(2)))

c_share = control["monster"].sum() / control["bev_lines"].sum()
print(f"\n  never-treated stores, Monster share overall: {c_share:.2%}")

pre = ev.filter(pl.col("evt") < 0)
post = ev.filter(pl.col("evt") >= 0)
if pre.height and post.height:
    a = pre["share"].mean()
    b = post["share"].mean()
    print(f"  treated  pre {a:.2%}  ->  post {b:.2%}   ({b - a:+.2%} points)")
