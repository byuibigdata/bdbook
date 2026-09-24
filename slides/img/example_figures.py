import polars as pl
from great_tables import GT, md, style, loc
from lets_plot import *
LetsPlot.setup_html()

places = pl.read_parquet("data/parquet/places.parquet")
patterns = pl.read_parquet("data/parquet/patterns.parquet")
j = places.join(patterns, on="placekey")

# --- Great Tables example: a small, well-labeled summary table
tbl = (j.group_by("location_name", "city", "region")
    .agg(pl.col("raw_visit_counts").sum().alias("visits"),
         pl.col("median_dwell").median().alias("dwell"))
    .sort("visits", descending=True).head(6))

gt = (GT(tbl)
    .tab_header(title="Busiest church buildings", subtitle="Utah and Georgia, Oct–Dec 2021")
    .cols_label(location_name="Building", city="City", region="State", visits="Visits", dwell="Median dwell (min)")
    .fmt_integer("visits")
    .fmt_number("dwell", decimals=0)
    .data_color(columns="visits", palette=["#F4F7F9", "#BDD3E5", "#426A8A"])
    .tab_style(style=style.text(weight="bold"), locations=loc.column_labels())
    .tab_source_note(md("Source: SafeGraph patterns, challenge data"))
    .tab_options(table_font_names="Open Sans", table_font_size="15px"))
open("gt.html", "w").write(gt.as_raw_html())

# --- Lets-Plot example: hourly rhythm, the kind of chart the challenge asks for
hours = (j.filter(pl.col("top_category") == "Religious Organizations")
    .select("region", "popularity_by_hour")
    .with_columns(hour=pl.int_ranges(0, pl.col("popularity_by_hour").list.len()))
    .explode("popularity_by_hour", "hour")
    .group_by("region", "hour")
    .agg(pl.col("popularity_by_hour").mean().alias("avg_visits"))
    .sort("hour"))

p = (ggplot(hours.to_dict(as_series=False), aes("hour", "avg_visits", color="region"))
    + geom_line(size=1.6)
    + scale_color_manual(values=["#D5773C", "#022F4A"], name="State")
    + labs(title="Religious buildings fill up at different hours",
           subtitle="Average visitors per hour, Utah vs Georgia",
           x="Hour of day", y="Average visitors", caption="Source: SafeGraph patterns")
    + theme_minimal()
    + theme(text=element_text(family="Open Sans"),
            plot_title=element_text(size=20, face="bold"))
    + ggsize(760, 440))
ggsave(p, "lp.html", path=".")
print("ok")
