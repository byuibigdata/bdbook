# %%
# https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
import polars as pl

# test = pl.read_csv("csv/test.csv", has_header=True)
pl.read_csv("csv/oil.csv", has_header=True, try_parse_dates=True).write_parquet("oil.parquet")
pl.read_csv("csv/stores.csv", has_header=True, try_parse_dates=True).write_parquet("stores.parquet")
pl.read_csv("csv/train.csv", has_header=True, try_parse_dates=True).write_parquet("transactions_daily.parquet")
pl.read_csv("csv/transactions.csv", has_header=True, try_parse_dates=True).write_parquet("transactions_store.parquet")
pl.read_csv("csv/holidays_events.csv", has_header=True, try_parse_dates=True).write_parquet("holiday_events.parquet")
# %%
