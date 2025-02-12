# %%
import polars as pl
from lets_plot import *
LetsPlot.setup_html()

oil = pl.read_parquet("oil.parquet")
stores = pl.read_parquet("stores.parquet")
daily = pl.read_parquet("transactions_daily.parquet")
tot_transactions = pl.read_parquet("transactions_store.parquet")
holiday = pl.read_parquet("holiday_events.parquet")
# %%

# Our data has daily transactions and we want to move it to weekly transactions. We have to make decisions about how we handle the aggregation of `sales` and `onpromotion`. For `sales` it makes sense to use `sum()` to aggregate the values for the week. The `onpromotion` column could be aggregated by a measure of center like `median()` or `mean()`. For our example we will focus on engineering the `sales` column.

weekly = daily.with_columns(
  pl.col("date").dt.week().alias("week_of_year"),
  pl.col("date").dt.year().alias("year")
  )\
  .group_by(['store_nbr', 'family', 'week_of_year', 'year'])\
  .agg(
    pl.col("sales").sum().alias("sales"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
    pl.col("date").count().alias("num_days_in_week"),
    pl.col('onpromotion').sum().alias("onpromotion_sum"),
    pl.col('onpromotion').median().alias("onpromotion_median"),
    pl.col('onpromotion').mean().alias("onpromotion_mean"))\
  .with_columns(
    pl.col("min_date").dt.weekday().alias("weekday").cast(pl.String()),
    (pl.col("min_date") - pl.duration(days=1)).alias("prediction_day"))
# %%
ggplot(weekly, aes(x="sales")) + geom_histogram()

# %%
# checking if weeks always start with with Monday.  They usually do. This looks to be what makes weeks different lengths a few times. Stops in week 33 of 2017
ggplot(weekly.unique(subset=['weekday', 'year', 'week_of_year']), aes(x="sales")) +\
  facet_wrap(facets='weekday', scales="free_y") +\
  geom_histogram()
# checking num of days in week
ggplot(weekly.unique(subset=['weekday', 'year', 'week_of_year']).with_columns(pl.col('year').cast(pl.String())), aes(x="num_days_in_week", fill='year')) +\
  geom_histogram() +\
  scale_y_continuous(trans='log10', breaks=[1, 2, 3, 4, 5, 6, 8, 10, 50, 100, 125])

weekly.unique(subset=['weekday', 'year', 'week_of_year']).group_by('year', 'week_of_year', 'num_days_in_week').len().filter(pl.col("num_days_in_week") != 7).sort('year', 'week_of_year')

# %%
# Since the goal of this post is not Feature Engineering for TS, we will try to keep this part as simple as possible. Let’s create two features which usually are used for time series. One step back in time, 1-lag(shift =1) and the difference between the number of purchase a week ago(W 1) and its previous week, means, two weeks ago (W2). After that, since the lag and diff result in having null in the dataset, see Fig. 4, we drop them. Therefore, when we look at the head of the data frame, it starts from week = 2.

weekly.sort('store_nbr', 'family','year', 'week_of_year')\
  .with_columns(
    pl.col('sales').over(['store_nbr', 'family']).shift(-1).alias("2_week_ahead_sales"),
    pl.col('sales').over(['store_nbr', 'family']).shift(1).alias("1_week_behind_sales"))\
  .rename({'sales':'1_week_ahead_sales'})\
  .select('store_nbr', 'family', 'week_of_year', 'year', 'prediction_day', 'min_date', 'max_date', 'num_days_in_week', '1_week_behind_sales', '1_week_ahead_sales', '2_week_ahead_sales')

# %%
# What is wrong with the date 2013-12-31?
# What is wrong with the date 2017-01-01?
# Can you fix this issue with weeks and years?
# Can you create the following table for 2017 SEAFOOD?
weekly_wr = weekly.sort('store_nbr', 'family','year', 'week_of_year')\
  .with_columns(
    pl.col('sales').over(['store_nbr', 'family']).shift(-1).alias("2_week_ahead_sales"),
    pl.col('sales').over(['store_nbr', 'family']).shift(1).alias("1_week_behind_sales"))\
  .rename({'sales':'1_week_ahead_sales'})\
  .select('store_nbr', 'family', 'week_of_year', 'year', 'prediction_day', 'min_date', 'max_date', 'num_days_in_week', '1_week_behind_sales', '1_week_ahead_sales', '2_week_ahead_sales')

print(weekly_wr\
  .filter((pl.col('family') == "SEAFOOD") & (pl.col('year') == 2016) & (pl.col("store_nbr") == 54) & (pl.col('week_of_year') >= 45))\
  .to_pandas()\
  .to_markdown())

# |    |   store_nbr | family   |   week_of_year |   year | prediction_day | min_date   | max_date   |   num_days_in_week |   1_week_behind_sales |   1_week_ahead_sales |   2_week_ahead_sales |
# |---:|------------:|:---------|---------------:|-------:|:---------------|:-----------|:-----------|-------------------:|----------------------:|---------------------:|---------------------:|
# |  0 |          54 | SEAFOOD  |             45 |   2016 | 2016-11-06     | 2016-11-07 | 2016-11-13 |                  7 |                     9 |                    3 |                   19 |
# |  1 |          54 | SEAFOOD  |             46 |   2016 | 2016-11-13     | 2016-11-14 | 2016-11-20 |                  7 |                     3 |                   19 |                    5 |
# |  2 |          54 | SEAFOOD  |             47 |   2016 | 2016-11-20     | 2016-11-21 | 2016-11-27 |                  7 |                    19 |                    5 |                    8 |
# |  3 |          54 | SEAFOOD  |             48 |   2016 | 2016-11-27     | 2016-11-28 | 2016-12-04 |                  7 |                     5 |                    8 |                   10 |
# |  4 |          54 | SEAFOOD  |             49 |   2016 | 2016-12-04     | 2016-12-05 | 2016-12-11 |                  7 |                     8 |                   10 |                   10 |
# |  5 |          54 | SEAFOOD  |             50 |   2016 | 2016-12-11     | 2016-12-12 | 2016-12-18 |                  7 |                    10 |                   10 |                    6 |
# |  6 |          54 | SEAFOOD  |             51 |   2016 | 2016-12-18     | 2016-12-19 | 2016-12-24 |                  6 |                    10 |                    6 |                    5 |
# |  7 |          54 | SEAFOOD  |             52 |   2016 | 2016-12-25     | 2016-12-26 | 2016-12-31 |                  6 |                     6 |                    5 |                    4 |
# |  8 |          54 | SEAFOOD  |             53 |   2016 | 2015-12-31     | 2016-01-01 | 2016-01-03 |                  3 |                     5 |                    4 |                   12 |
# %%
# Which column would be our `y` value or `target` if we were predicting what we will sell in the next seven days?
# Can you build a new `y` that is the lagged difference? The column would be the difference in sales from the week we just completed to the sales seven days in the future.
# Can you build a new `feature` that is the difference in percent of sales for the year comparing the most recently completed week with it's sister week in the previous year?

weekly_wr\
  .with_columns(
    (pl.col('1_week_behind_sales') - pl.col('1_week_ahead_sales')).alias('1_week_ahead_change_previous')
  )

prop_previous_year = weekly_wr\
  .with_columns(
    pl.col('1_week_behind_sales').sum().over('store_nbr', 'family', 'year').alias('year_total'))\
  .with_columns((pl.col('1_week_behind_sales') / pl.col('year_total')).alias('prop_year'))\
  .select('store_nbr', 'family', 'week_of_year', 'year', 'prop_year')\
  .pivot(on='year', index=['store_nbr', 'family', 'week_of_year'])\
  .with_columns(
    (pl.col('2017') - pl.col('2016')).alias('2017'),
    (pl.col('2016') - pl.col('2015')).alias('2016'),
    (pl.col('2015') - pl.col('2014')).alias('2015'),
    (pl.col('2014') - pl.col('2013')).alias('2014'))\
  .drop("2013")\
  .unpivot(index=['store_nbr', 'family', 'week_of_year'])\
  .drop_nulls()

#  .filter((pl.col('family') == "SEAFOOD") & (pl.col('year') == 2016) & (pl.col("store_nbr") == 54) & (pl.col('week_of_year') >= 45))\


# %%
ggplot(prop_previous_year, aes(x='value')) +\
  geom_histogram(bins=400000) +\
  scale_x_continuous(limits=(-.05, .05)) +\
  coord_cartesian(ylim=(0, 4000))
# %%
ggplot(prop_previous_year, aes(x='value')) +\
  geom_histogram(bins=400000) +\
  scale_x_continuous(limits=(-.05, .05)) +\
  coord_cartesian(ylim=(0, 200)) + facet_wrap(facets='store_nbr')
# %%
ft_table = weekly_wr.select(['store_nbr', 'family', 'week_of_year', 'prediction_day', '1_week_behind_sales', '1_week_ahead_sales'])\
  .with_columns((pl.col('1_week_behind_sales') - pl.col('1_week_ahead_sales')).alias('1_week_ahead_change_previous'))\
  .join(prop_previous_year.rename({'value':'pdp_yearweek'}), on=['store_nbr', 'family', 'week_of_year'], how='left')
# %%

previous_week = ggplot(ft_table.filter(pl.col('family') == 'SEAFOOD'), aes(x="1_week_behind_sales", y='1_week_ahead_sales', color='store_nbr')) +\
  geom_point() +\
  facet_wrap(facets='store_nbr', scales="free") +\
  geom_smooth()

ggsave(previous_week, filename="previous_week.html")
# %%
previous_week_diff = ggplot(ft_table.filter(pl.col('family') == 'SEAFOOD'), aes(x="pdp_yearweek", y='1_week_ahead_sales', color='store_nbr')) +\
  geom_point() +\
  facet_wrap(facets='store_nbr', scales="free") +\
  geom_smooth()

ggsave(previous_week_diff, filename="previous_week_diff.html")

# %%
previous_year_percent = ggplot(ft_table.sample(n=10000), aes(x="pdp_yearweek", y='1_week_ahead_sales')) +\
  geom_point() +\
  scale_x_continuous(limits=(-.05, .05)) +\
  facet_wrap(facets='store_nbr', scales="free") +\
  geom_smooth()

ggsave(previous_year_percent, filename="previous_year_percent.html")

# %%
previous_year_percent_change = ggplot(ft_table.filter(pl.col('family') == 'SEAFOOD'), aes(x="pdp_yearweek", y='1_week_ahead_change_previous', color='store_nbr')) +\
  geom_point() +\
  facet_wrap(facets='store_nbr', scales="free") +\
  geom_smooth()

ggsave(previous_year_percent_change, filename="previous_year_percent_change.html")

# %%
