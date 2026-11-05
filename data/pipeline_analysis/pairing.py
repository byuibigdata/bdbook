"""Pairing offers: the second item, not the one they came for.

The anchor pipeline (`serve.py`) predicts what a customer buys anyway. That
makes it the wrong thing to discount -- a promotion on the item they had
already decided to buy gives away margin and changes no behaviour. It is the
one offer guaranteed to lose.

This module asks the other question. Given the category they are going to buy
anyway, what SECOND category could plausibly join the basket, and which SKU
inside it?

Three measured quantities, kept separate on purpose:

    basket_lift    P(B | A in basket) / P(B).  Basket-level co-occurrence.
    cohort_index   per-basket rate of B among A-regulars, divided by the rate
                   among EQUALLY FREQUENT other shoppers. A ratio.
    excess         the same comparison as a DIFFERENCE: cohort_rate minus
                   other_rate, in percentage points of attach.
    headroom       1 - this shopper's own observed rate of B. Discounting
                   what they already buy every visit is waste.

**Gate on the index, rank on the excess.** Both are needed and neither works
alone:

  - Raw co-occurrence volume picks the popular item. Candy is 18.2% of every
    second item in a drink basket and looks like the obvious pair, but its
    index is 1.07 -- drink regulars buy it barely more than anyone else. It
    is not a complement, it is just everywhere.
  - The index alone picks tiny categories. Other Dairy indexes 1.32 on a
    0.55% attach rate; the ratio is flattering and the volume is noise.
  - The excess, cohort_rate - other_rate, is in attach percentage points, so
    it ranks by how much extra business the pairing is actually associated
    with. Salty Snacks 2.8pp, Alternative Snacks 2.6pp, Sweet Snacks 2.0pp,
    Candy 0.8pp -- which puts candy where it belongs.

    uv run --with polars --with numpy python pairing.py --build
    uv run --with polars --with numpy python pairing.py --anchor "Salty Snacks"
    uv run --with polars --with scikit-learn --with implicit --with joblib \\
        python pairing.py --demo

Alcohol is excluded on both sides. A beer anchor returns no pair offer and
alcohol is never the second item, which is `config.NOT_PAIRABLE` rather than
a measurement -- though the measurement agrees: beer's only complement above
1.0 is Liquor, at 1.08.

WHAT THIS IS NOT: causal. Co-occurrence says these things appear together,
not that a discount on B causes a B purchase. Measuring that needs a holdout
and it does not exist. See the holdout note in the README.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

import config as C
from scoring import slug

PAIR_TABLE = "pair_affinity.parquet"


# ------------------------------------------------------------------- build
def _baskets() -> pl.DataFrame:
    """Purchase baskets from the training period, as category sets.

    Training period only. Pair strengths measured on the test period would
    be measured on the same fills we later score, which is the same leak the
    rest of the pipeline is careful about.
    """
    v = pl.read_parquet(
        C.ART / "visit.parquet",
        columns=["fuel_tx", "shopper_id", "ts", "cats", "bought"])
    return v.filter(
        pl.col("bought")
        & (pl.col("ts") < pl.lit(C.TRAIN_CUTOFF).str.to_datetime())
    ).select("fuel_tx", "shopper_id", "cats")


def _basket_pairs(b: pl.DataFrame) -> pl.DataFrame:
    """Basket-level co-occurrence: support, confidence, lift."""
    n = b.height
    ex = b.select("fuel_tx", "cats").explode("cats").rename({"cats": "anchor"})
    marg = (ex.group_by("anchor").len().rename({"len": "n_anchor"})
            .with_columns(p_anchor=pl.col("n_anchor") / n))
    pairs = (
        ex.join(ex.rename({"anchor": "pair"}), on="fuel_tx")
        .filter(pl.col("anchor") != pl.col("pair"))
        .group_by(["anchor", "pair"]).len().rename({"len": "n_both"}))
    return (
        pairs
        .join(marg.select("anchor", "n_anchor"), on="anchor")
        .join(marg.select(pl.col("anchor").alias("pair"),
                          pl.col("p_anchor").alias("p_pair")), on="pair")
        .with_columns(confidence=pl.col("n_both") / pl.col("n_anchor"))
        .with_columns(basket_lift=pl.col("confidence") / pl.col("p_pair")))


def _cohort_index(b: pl.DataFrame) -> pl.DataFrame:
    """Per-basket rate of B among A-regulars vs equally frequent others.

    The comparison group is the whole point. Every cohort here is defined by
    "at least PAIR_MIN_VISITS baskets", so every cohort is a cohort of heavy
    shoppers and indexes high against the general population on nearly
    everything. Holding visit count constant on both sides divides that out
    and leaves the thing we actually want: do people who buy A buy more B
    than people who shop just as often and do not buy A?
    """
    visits = b.group_by("shopper_id").agg(n=pl.len())
    elig = visits.filter(pl.col("n") >= C.PAIR_MIN_VISITS)
    counts = (b.select("shopper_id", "cats").explode("cats")
              .group_by(["shopper_id", "cats"]).len()
              .rename({"cats": "cat", "len": "n_cat"}))
    elig_counts = counts.join(elig, on="shopper_id", how="inner")
    rates = elig_counts.with_columns(rate=pl.col("n_cat") / pl.col("n"))

    out = []
    anchors = rates["cat"].unique().sort().to_list()
    for a in anchors:
        members = rates.filter((pl.col("cat") == a)
                               & (pl.col("rate") > C.PAIR_DOMINANT))["shopper_id"]
        if members.len() < 50:
            continue
        inside = elig.filter(pl.col("shopper_id").is_in(members.implode()))
        outside = elig.filter(~pl.col("shopper_id").is_in(members.implode()))
        n_in, n_out = inside["n"].sum(), outside["n"].sum()
        if not n_in or not n_out:
            continue
        tot = (elig_counts.with_columns(
            grp=pl.when(pl.col("shopper_id").is_in(members.implode()))
            .then(pl.lit("in")).otherwise(pl.lit("out")))
            .group_by(["cat", "grp"]).agg(k=pl.col("n_cat").sum())
            .pivot(on="grp", index="cat", values="k").fill_null(0))
        if "in" not in tot.columns or "out" not in tot.columns:
            continue
        out.append(
            tot.filter(pl.col("cat") != a)
            .with_columns(
                anchor=pl.lit(a),
                cohort_shoppers=pl.lit(members.len(), dtype=pl.Int64),
                cohort_baskets=pl.lit(int(n_in), dtype=pl.Int64),
                cohort_support=pl.col("in"),
                cohort_rate=pl.col("in") / n_in,
                other_rate=pl.col("out") / n_out)
            .with_columns(cohort_index=pl.when(pl.col("other_rate") > 0)
                          .then(pl.col("cohort_rate") / pl.col("other_rate"))
                          .otherwise(None))
            .rename({"cat": "pair"})
            .select("anchor", "pair", "cohort_shoppers", "cohort_baskets",
                    "cohort_support", "cohort_rate", "other_rate",
                    "cohort_index"))
    return pl.concat(out) if out else pl.DataFrame()


def build_pair_table(save: bool = True) -> pl.DataFrame:
    b = _baskets()
    t = (_basket_pairs(b)
         .join(_cohort_index(b), on=["anchor", "pair"], how="full",
               coalesce=True)
         .sort(["anchor", "cohort_index"], descending=[False, True]))
    if save:
        t.write_parquet(C.ART / PAIR_TABLE)
    return t


# The second-item struct published in customer_scores.parquet. No p_category
# and no p_buys: the pair category has no Category Relevance model, so there
# is no P(category) for it and any product using one would be invented.
# `cohort_index` and `excess` are the pair's evidence instead.
SECOND_SKU_FIELDS = ["rank", "gtin", "brand", "category", "affinity",
                     "p_sku", "cohort_index", "excess", "headroom"]


def pair_candidates(pairs: pl.DataFrame, anchor: str) -> pl.DataFrame:
    """Pair categories for this anchor that clear every gate, best first.

    One function so the live engine and the batch column cannot drift. The
    gates are: enough cohort support to be a rate rather than a rumour, an
    index above PAIR_MIN_INDEX so it is a complement rather than a popular
    item, and not an excluded category. Ordered by excess, the attach
    difference in percentage points -- see the module docstring on why the
    ratio gates and the difference ranks.
    """
    if anchor in C.NOT_PAIRABLE:
        return pairs.clear()
    return (pairs.filter(
        (pl.col("anchor") == anchor)
        & (pl.col("cohort_support") >= C.PAIR_MIN_SUPPORT)
        & (pl.col("cohort_index") >= C.PAIR_MIN_INDEX)
        & ~pl.col("pair").is_in(C.NOT_OFFERABLE + C.NOT_PAIRABLE))
        .with_columns(excess=pl.col("cohort_rate") - pl.col("other_rate"))
        .sort("excess", descending=True))


def own_shares(visit: pl.DataFrame, cats: list[str]) -> pl.DataFrame:
    """Per-visit prior rate of each category, strictly before that visit.

    The batch twin of `PairEngine._own_rates`, and the same definition: the
    denominator is the shopper's prior PURCHASE baskets, not their fuel
    visits, because the pair rates it is compared against are per-basket.

    Only the handful of categories that survive the gates are computed. All
    32 would be 20x the work to answer a question nobody asks.
    """
    v = (visit.select("fuel_tx", "shopper_id", "ts", "cats", "bought")
         .sort("shopper_id", "ts", "fuel_tx"))
    b = pl.col("bought").cast(pl.Int32)
    over = pl.col("shopper_id")
    exprs = [(b.cum_sum().over(over) - b).alias("_prior_n")]
    for c in cats:
        hit = (pl.col("bought")
               & pl.col("cats").list.contains(c).fill_null(False)).cast(pl.Int32)
        exprs.append((hit.cum_sum().over(over) - hit).alias(f"_h_{slug(c)}"))
    v = v.with_columns(exprs)
    return v.select(
        "fuel_tx",
        *[pl.when(pl.col("_prior_n") > 0)
          .then(pl.col(f"_h_{slug(c)}") / pl.col("_prior_n"))
          .otherwise(0.0).alias(slug(c)) for c in cats])


class BatchPairer:
    """The second-item column, for a whole table of customer-moments.

    Same two stages as the anchor pipeline, one level out: pick the pair
    CATEGORY from the complement table, then the SKU inside it from ALS.
    """

    def __init__(self, ranker, pairs: pl.DataFrame,
                 catalog_cats: set[str]) -> None:
        self.ranker, self.pairs, self.catalog_cats = ranker, pairs, catalog_cats

    def candidate_cats(self, anchors: list[str]) -> list[str]:
        """Every pair category reachable from any anchor in this table."""
        out: set[str] = set()
        for a in set(anchors):
            if a:
                out |= set(pair_candidates(self.pairs, a)["pair"].to_list())
        return sorted(out)

    def choose(self, anchors: list[str], own: pl.DataFrame
               ) -> tuple[list[str | None], np.ndarray, np.ndarray, np.ndarray]:
        """Pick each row's pair category by excess x headroom."""
        n = len(anchors)
        cats: list[str | None] = [None] * n
        idx = np.zeros(n); exc = np.zeros(n); hr = np.zeros(n)
        anch = np.asarray(anchors, dtype=object)
        for a in {x for x in anchors if x}:
            cand = pair_candidates(self.pairs, a)
            if cand.height == 0:
                continue
            rows = np.flatnonzero(anch == a)
            names = cand["pair"].to_list()
            e = cand["excess"].to_numpy()
            ci = cand["cohort_index"].to_numpy()
            # headroom = 1 - what this shopper already does. Discounting what
            # they buy every visit is the one offer certain to change nothing.
            H = np.column_stack([
                1.0 - own[slug(c)].to_numpy()[rows] for c in names])
            H = np.clip(H, 0.0, 1.0)
            best = np.argmax(H * e[None, :], axis=1)
            for j, r in enumerate(rows):
                k = int(best[j])
                cats[r] = names[k]
                idx[r], exc[r], hr[r] = ci[k], e[k], H[j, k]
        return cats, idx, exc, hr

    def skus(self, shoppers: list[str], dayparts: list[str],
             cats: list[str | None], idx, exc, hr, top: int = 5
             ) -> list[list[dict]]:
        """Rank SKUs inside each row's chosen pair category."""
        from scoring import P_FLOOR
        out: list[list[dict]] = [[] for _ in cats]
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cats):
            if c and c in self.catalog_cats:
                groups.setdefault(c, []).append(i)
        for cat, rows in groups.items():
            sh = [shoppers[i] for i in rows]
            dp = [dayparts[i] for i in rows]
            # The pair category has no relevance model, so it is forced in
            # at certainty purely to make it the one category the ranker may
            # choose. That is why p_category never leaves this function.
            props = {cat: np.full(len(rows), 1.0 - P_FLOOR)}
            picks = self.ranker.rank(sh, dp, props, top=top)
            for j, i in enumerate(rows):
                out[i] = [{"rank": c["rank"], "gtin": c["gtin"],
                           "brand": c["brand"], "category": c["category"],
                           "affinity": round(c["affinity"], 6),
                           "p_sku": round(c["p_sku"], 6),
                           "cohort_index": round(float(idx[i]), 6),
                           "excess": round(float(exc[i]), 6),
                           "headroom": round(float(hr[i]), 6)}
                          for c in picks[j]]
        return out


# ----------------------------------------------------------------- serving
@dataclass
class PairOffer:
    anchor_category: str
    anchor_brand: str | None       # what they buy anyway -- do NOT discount
    pair_category: str
    pair_brand: str | None
    pair_gtin: str | None
    cohort_index: float            # complementarity, exposure controlled
    basket_lift: float             # basket-level co-occurrence
    cohort_rate: float             # how often people like them buy it
    other_rate: float              # ...and equally frequent non-anchor folk
    excess: float                  # cohort_rate - other_rate, in attach pp
    own_rate: float                # how often THIS shopper already does
    headroom: float                # 1 - own_rate
    score: float                   # excess x headroom
    in_catalog: bool               # is there a SKU we could actually show?

    def __str__(self) -> str:
        sku = f"{self.pair_brand}" if self.pair_brand else "(no SKU in catalog)"
        return (f"anchor {self.anchor_category[:28]:<30} "
                f"pair {self.pair_category[:26]:<28} {sku:<16} "
                f"index {self.cohort_index:.2f}  headroom {self.headroom:.0%}")


class PairEngine:
    """Recommends a second item, given the anchor the main pipeline predicts."""

    def __init__(self, engine=None) -> None:
        from serve import OfferEngine
        self.engine = engine or OfferEngine()
        path = C.ART / PAIR_TABLE
        if not path.exists():
            raise SystemExit(
                f"{path} missing -- run: python pairing.py --build")
        self.pairs = pl.read_parquet(path)
        self.catalog_cats = set(
            self.engine.catalog.filter(
                pl.col("in_stock") & pl.col("offerable"))["category"].to_list())

    def _own_rates(self, shopper_id: str, before: datetime) -> dict[str, float]:
        """This shopper's own per-basket rate for every category, before now.

        Measured, not modelled -- and over all 32 categories, not just the
        six the relevance models cover. The pair candidate is usually outside
        those six.
        """
        prior = self.engine.visit.filter(
            (pl.col("shopper_id") == shopper_id)
            & (pl.col("ts") < before) & pl.col("bought"))
        n = prior.height
        if n == 0:
            return {}
        ex = prior.select("cats").explode("cats").drop_nulls()
        return {r["cats"]: r["len"] / n
                for r in ex.group_by("cats").len().iter_rows(named=True)}

    def recommend(self, shopper_id: str, when: str | datetime,
                  fuel_grade: str, store_id: int | None = None,
                  top: int = 5,
                  force_anchor: str | None = None
                  ) -> tuple[list[PairOffer], object]:
        """Pair offers for this fill, best first.

        `force_anchor` overrides stage one and asks "what would we pair with
        THIS category" instead. It exists for validation and for exploring an
        anchor the model did not pick; it does not change the anchor offer.
        """
        ts = when if isinstance(when, datetime) else datetime.fromisoformat(when)
        anchor = self.engine.decide(shopper_id, ts, fuel_grade,
                                    store_id=store_id, top=1)
        cat = force_anchor or anchor.category
        if cat is None:
            return [], anchor
        # An alcohol anchor takes no second item. See config.NOT_PAIRABLE.
        if cat in C.NOT_PAIRABLE:
            return [], anchor
        own = self._own_rates(shopper_id, ts)

        cands = pair_candidates(self.pairs, cat)

        offers = []
        for r in cands.iter_rows(named=True):
            own_rate = own.get(r["pair"], 0.0)
            headroom = max(0.0, 1.0 - own_rate)
            # The excess attach associated with the anchor, times the room
            # left to grow it. Ranking on cohort_rate instead would rank on
            # popularity and put Candy first, which is the error the index
            # exists to catch. Both terms are measured; neither is a causal
            # uplift, because nothing here is an experiment.
            excess = float(r["cohort_rate"]) - float(r["other_rate"])
            score = excess * headroom
            gtin = brand = None
            if r["pair"] in self.catalog_cats:
                best = self._best_sku(shopper_id, anchor.when, r["pair"])
                if best:
                    gtin, brand = best
            offers.append(PairOffer(
                anchor_category=cat,
                anchor_brand=(anchor.ranked[0].brand
                              if anchor.ranked and not force_anchor else None),
                pair_category=r["pair"], pair_brand=brand, pair_gtin=gtin,
                cohort_index=float(r["cohort_index"]),
                basket_lift=float(r["basket_lift"] or 0.0),
                cohort_rate=float(r["cohort_rate"]),
                other_rate=float(r["other_rate"]), excess=excess,
                own_rate=own_rate, headroom=headroom, score=score,
                in_catalog=r["pair"] in self.catalog_cats))
        offers.sort(key=lambda o: o.score, reverse=True)
        return offers[:top], anchor

    def _best_sku(self, shopper_id: str, when: datetime,
                  category: str) -> tuple[str, str] | None:
        """The SKU inside the pair category, by ALS affinity for this shopper.

        Reuses the anchor pipeline's ranker, with the pair category forced in
        place of the one stage one chose.
        """
        from scoring import P_FLOOR
        import numpy as np
        rk = self.engine.ranker
        dp = C.daypart_of(when.hour)
        props = {category: np.array([1.0 - P_FLOOR])}
        picks = rk.rank([shopper_id], [dp], props, top=1)[0]
        if not picks:
            return None
        return picks[0]["gtin"], picks[0]["brand"]


# --------------------------------------------------------------------- CLI
def show_anchor(t: pl.DataFrame, anchor: str, n: int = 12) -> None:
    sub = t.filter((pl.col("anchor") == anchor)
                   & (pl.col("cohort_support") >= C.PAIR_MIN_SUPPORT)
                   & ~pl.col("pair").is_in(C.NOT_PAIRABLE))
    if anchor in C.NOT_PAIRABLE:
        print(f"\nANCHOR  {anchor}   [not pairable]")
        print("Alcohol is excluded from pairing on both sides; see "
              "config.NOT_PAIRABLE.")
        return
    if sub.height == 0:
        print(f"no pairs with support for {anchor!r}. Anchors available:")
        for a in t["anchor"].unique().sort().to_list():
            print(f"   {a}")
        return
    row = sub.row(0, named=True)
    print(f"\nANCHOR  {anchor}")
    print(f"cohort  {row['cohort_shoppers']:,} shoppers "
          f"(>={C.PAIR_MIN_VISITS} baskets, >{C.PAIR_DOMINANT:.0%} of them "
          f"holding it), {row['cohort_baskets']:,} baskets\n")
    print(f"  {'pair category':<34}{'index':>7}{'lift':>7}"
          f"{'their rate':>12}{'others':>9}{'2nd-item share':>16}")
    tot = sub["n_both"].sum()
    for r in sub.sort("cohort_index", descending=True).head(n).iter_rows(named=True):
        mark = "" if r["cohort_index"] >= C.PAIR_MIN_INDEX else "   [not a complement]"
        print(f"  {r['pair'][:33]:<34}{r['cohort_index']:>7.2f}"
              f"{(r['basket_lift'] or 0):>7.2f}{r['cohort_rate']:>12.4f}"
              f"{r['other_rate']:>9.4f}{(r['n_both'] / tot):>15.1%}{mark}")
    print("\n  index  = their rate / the rate among equally frequent others")
    print("  lift   = basket co-occurrence over base rate")
    print("  share  = this pair's share of all second items in those baskets")
    print("\n  Gate on index, rank on excess (their rate - others, in pp).")
    print("  Share alone picks the popular item; index alone picks the tiny one.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build the pair table")
    ap.add_argument("--anchor", help="print complements for this category")
    ap.add_argument("--shopper")
    ap.add_argument("--time", default="2026-06-15 07:30")
    ap.add_argument("--grade", default="REGULAR")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.build:
        t = build_pair_table()
        print(f"wrote {C.ART / PAIR_TABLE}  ({t.height:,} pairs, "
              f"{t['anchor'].n_unique()} anchors)")
        show_anchor(t, C.TARGET_CATEGORIES[0])
        return

    if args.anchor:
        show_anchor(pl.read_parquet(C.ART / PAIR_TABLE), args.anchor)
        return

    eng = PairEngine()
    shopper = args.shopper
    if args.demo or not shopper:
        shopper = (eng.engine.visit.group_by("shopper_id")
                   .agg(n=pl.len(), k=pl.col("bought").sum())
                   .filter((pl.col("n") >= 8) & (pl.col("k") >= 5))
                   .sort(["k", "shopper_id"], descending=[True, False])
                   .row(0)[0])
        print(f"(demo shopper {shopper})")
    when = datetime.fromisoformat(args.time.replace(" ", "T"))

    offers, anchor = eng.recommend(shopper, when, args.grade)
    print(f"\nanchor      {anchor.category}")
    if anchor.ranked:
        a = anchor.ranked[0]
        print(f"            best SKU {a.brand}   P(buys) {a.p_buy:.4f}")
    print("            this is what they buy ANYWAY -- do not discount it\n")
    if not offers:
        if anchor.category in C.NOT_PAIRABLE:
            print(f"{anchor.category} takes no pair offer -- alcohol is out of")
            print("the pairing question on both sides (config.NOT_PAIRABLE).")
            return
        print("no pair clears the complement test. Showing a second item here")
        print("would be offering something these customers buy LESS than average.")
        return
    print(f"  {'pair':<30}{'SKU':<16}{'index':>7}{'excess':>9}"
          f"{'their rate':>12}{'headroom':>10}{'score':>9}")
    for o in offers:
        sku = o.pair_brand or "— not in catalog"
        print(f"  {o.pair_category[:29]:<30}{sku[:15]:<16}{o.cohort_index:>7.2f}"
              f"{o.excess * 100:>8.2f}pp{o.cohort_rate:>12.4f}"
              f"{o.headroom:>10.0%}{o.score:>9.4f}")
    best = offers[0]
    print(f"\n  PAIR OFFER: {best.pair_brand or best.pair_category}"
          f"  ({best.pair_category})")
    if not best.in_catalog:
        print("  WARNING: no SKU for this category in offer_catalog.csv --")
        print("  add it to config.SKU_MODEL_CATEGORIES and rebuild.")
    print("\n  Co-occurrence is not causation. Nothing here measures what a")
    print("  discount CHANGES; it measures what already goes together.")


if __name__ == "__main__":
    main()
