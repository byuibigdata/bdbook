# %%
import polars as pl
dat = pl.read_parquet('draft_vietnam.parquet')

# %%
dat.write_csv("temp.csv")
# %%
