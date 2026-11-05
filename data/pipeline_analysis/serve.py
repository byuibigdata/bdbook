"""Answer the pump-screen question for one customer.

Give it a shopper, a time and a fuel grade; it returns the offer to show, or
tells you to show nothing.

The answer comes in two stages. Category Relevance picks the category -- the
most likely one we are allowed to show, on probability alone. Then SKU
Affinity and the margin pick the item inside that category. A SKU from any
other category cannot appear, however profitable it is.

    uv run --with polars --with scikit-learn --with joblib python serve.py \
        --shopper 0a1b2c... --time "2026-06-15 07:30" --grade REGULAR

    python serve.py --demo          # picks a real shopper for you

As a library:

    from serve import OfferEngine
    eng = OfferEngine()
    eng.decide(shopper_id="...", when="2026-06-15 07:30", fuel_grade="REGULAR")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import polars as pl

import config as C
import scoring
from scoring import SkuRanker

# One source of truth for the probability floor, shared with the batch path.
P_FLOOR = scoring.P_FLOOR


def daypart_of(ts: datetime) -> str:
    return C.daypart_of(ts.hour)


@dataclass
class Offer:
    gtin: str
    brand: str
    category: str
    p_category: float
    p_sku: float
    unit_margin: float
    discount: float
    funded_share: float      # share of the discount the BRAND pays
    our_cost: float          # discount x (1 - funded_share)
    expected_value: float

    @property
    def p_buy(self) -> float:
        return self.p_category * self.p_sku

    def min_funded_share(self, rel_uplift: float) -> float:
        """Least the brand must fund for this offer to beat showing nothing.

        Showing the offer wins when p1 (m - d(1-f)) > p0 m. Writing the
        uplift as relative, p1 = p0 (1 + r), p0 cancels and the floor is

            f_min = 1 - (m / d) x r / (1 + r)

        so the answer does not depend on how likely this customer was to
        buy. That decides how much the deal is worth, not who pays for it.
        `r` is an assumption: without a holdout we cannot measure it.

        0 means the offer pays for itself. Above 1 means no amount of brand
        money fixes it -- the discount destroys more margin than the extra
        volume returns.
        """
        if self.discount <= 0 or rel_uplift <= 0:
            return 1.0
        return 1.0 - (self.unit_margin / self.discount) * (
            rel_uplift / (1.0 + rel_uplift))

    def __str__(self) -> str:
        if self.funded_share >= 0.999:
            funder = "brand funds all"
        elif self.funded_share <= 0.001:
            funder = "we fund all"
        else:
            funder = f"split {self.funded_share:.0%}/{1 - self.funded_share:.0%}"
        return (f"{self.brand or '?':<16} {self.category[:26]:<26} "
                f"EV ${self.expected_value:7.4f}  "
                f"P(cat)={self.p_category:.4f} "
                f"P(sku)={self.p_sku:.4f}  {funder}")


@dataclass
class Decision:
    shopper_id: str
    when: datetime
    fuel_grade: str
    store_id: int
    known_shopper: bool
    prior_attach: float          # -1 when we have never seen them
    ranked: list[Offer] = field(default_factory=list)

    @property
    def category(self) -> str | None:
        """The category stage one chose. Every ranked offer sits inside it."""
        return self.ranked[0].category if self.ranked else None

    @property
    def show(self) -> Offer | None:
        """The offer worth showing, or None. Showing nothing is a real answer."""
        if self.ranked and self.ranked[0].expected_value >= C.MIN_EXPECTED_VALUE:
            return self.ranked[0]
        return None


class OfferEngine:
    """Loads the artifacts once and scores a fill in milliseconds."""

    def __init__(self) -> None:
        rel = joblib.load(C.ART / "model_category_relevance.joblib")
        aff = joblib.load(C.ART / "model_sku_affinity.joblib")
        self.relevance = rel["models"]          # category -> calibrated GBT
        self.cols = rel["cols"]
        self.cat_maps = rel["cat_maps"]
        self.per_cat = rel["per_cat"]
        c = aff
        self.als, self.shopper_ix = c["als"], c["shopper_ix"]
        self.items, self.popularity, self.matrix = c["items"], c["popularity"], c["matrix"]

        # GTINs are zero-padded to 14 digits; let Polars infer the type and
        # it reads them as integers and silently drops the leading zeros,
        # after which nothing joins.
        self.catalog = pl.read_csv(
            C.ART / "offer_catalog.csv",
            schema_overrides={"gtin": pl.Utf8},
        )
        self.visit = pl.read_parquet(
            C.ART / "visit.parquet",
            columns=["shopper_id", "store_id", "ts", "cats", "bought", "fuel_spend"],
        )
        self.hist = pl.read_parquet(C.ART / "feat_history.parquet")
        # The ranker is the shared core. `score_table.py` builds the same
        # object from the same two artifacts, so the two paths cannot drift.
        self.ranker = SkuRanker(aff, self.catalog)
        # Median fill, used when the caller does not supply one.
        self.default_spend = float(self.visit["fuel_spend"].median())

    # -- features ---------------------------------------------------------
    def _home_store(self, shopper_id: str) -> int | None:
        hit = self.visit.filter(pl.col("shopper_id") == shopper_id)
        if hit.height == 0:
            return None
        return int(hit.group_by("store_id").len().sort("len", descending=True)
                   .row(0)[0])

    def _history(self, shopper_id: str, before: datetime) -> dict:
        """Point-in-time history: only visits strictly before this one."""
        prior = self.visit.filter(
            (pl.col("shopper_id") == shopper_id) & (pl.col("ts") < before)
        )
        out = {}
        n = prior.height
        # The two that matter most: how often we have seen them, and what
        # share of those visits ended with them walking inside. The count is
        # worth +0.004 AUC; the share is worth +0.109.
        out["prior_n"] = n
        out["prior_attach"] = float(prior["bought"].mean()) if n else -1.0
        for cat in C.TARGET_CATEGORIES:
            hits = (prior["cats"].list.contains(cat).fill_null(False).sum()
                    if n else 0)
            out[f"prior_n_{cat}"] = n
            out[f"bought_before_{cat}"] = int(hits > 0)
            out[f"prior_share_{cat}"] = (hits / n) if n else 0.0
        return out

    # -- the two models ---------------------------------------------------
    def _row(self, shopper_id: str, ts, fuel_grade: str, store: int,
             spend: float) -> tuple[pl.DataFrame, dict]:
        """The one-row feature frame for this fill, plus the raw history.

        Built in exactly the shape `score_table.py` hands to the batch path,
        so the shared core sees no difference between a live fill and a row
        of the test period.
        """
        hist = self._history(shopper_id, ts)
        row = {
            "daypart": daypart_of(ts), "fuel_grade": fuel_grade,
            "store_id": str(store), "hour": ts.hour,
            "weekday": ts.isoweekday(), "month": ts.month,
            "fuel_spend": float(spend),
            **{k: v for k, v in hist.items()},
        }
        frame = pl.DataFrame([{c: row.get(c, 0) for c in
                               set(self.cols) | set(self.cat_maps)}])
        return frame, hist

    def category_probs(self, shopper_id: str, when, fuel_grade: str,
                       store_id: int | None = None,
                       fuel_spend: float | None = None) -> dict[str, float]:
        """P(buys this category on this fill), per category. Unconditional.

        There is no separate "will they come inside" model. An earlier
        version multiplied a P(inside) model into this one; measured against
        a single direct model the two matched to three decimals, so the extra
        stage was dropped. Coming inside is already inside these numbers.
        """
        ts = when if isinstance(when, datetime) else datetime.fromisoformat(when)
        store, _ = self._resolve_store(shopper_id, store_id)
        spend = fuel_spend if fuel_spend is not None else self.default_spend
        frame, _ = self._row(shopper_id, ts, fuel_grade, store, spend)
        X = scoring.design_matrix(frame, self.cat_maps, self.cols)
        return {cat: float(p[0])
                for cat, p in scoring.category_probs(self.relevance, X).items()}

    def prob_entering(self, shopper_id: str, when, fuel_grade: str,
                      store_id: int | None = None,
                      fuel_spend: float | None = None) -> float:
        """P(enters the store), recovered from the six categories.

        The independence product, not a model. See `scoring.prob_entering`.
        """
        p = self.category_probs(shopper_id, when, fuel_grade, store_id, fuel_spend)
        return float(scoring.prob_entering(
            {k: np.array([v]) for k, v in p.items()})[0])

    def _resolve_store(self, shopper_id: str,
                       store_id: int | None) -> tuple[int, bool]:
        """The store to score at, and whether we actually knew it."""
        if store_id is not None:
            return int(store_id), True
        store = self._home_store(shopper_id)
        if store is not None:
            return store, True
        busiest = int(self.visit.group_by("store_id").len()
                      .sort("len", descending=True).row(0)[0])
        return busiest, False

    # -- the decision -----------------------------------------------------
    def decide(self, shopper_id: str, when: str | datetime,
               fuel_grade: str, store_id: int | None = None,
               fuel_spend: float | None = None, top: int = 5) -> Decision:
        ts = when if isinstance(when, datetime) else datetime.fromisoformat(when)
        store, known = self._resolve_store(shopper_id, store_id)
        spend = fuel_spend if fuel_spend is not None else self.default_spend

        frame, hist = self._row(shopper_id, ts, fuel_grade, store, spend)
        X = scoring.design_matrix(frame, self.cat_maps, self.cols)
        props = scoring.category_probs(self.relevance, X)

        # The N = 1 case of the batch ranker: stage one picks the category by
        # relevance, stage two ranks that category's SKUs by expected value.
        # Same normalisation and same tie-breaking as the table on disk.
        picks = self.ranker.rank([shopper_id], [daypart_of(ts)], props, top=top)[0]
        offers = [Offer(c["gtin"], c["brand"], c["category"], c["p_category"],
                        c["p_sku"], c["unit_margin"], c["discount"],
                        c["funded_share"], c["our_cost"], c["expected_value"])
                  for c in picks]
        return Decision(shopper_id, ts, fuel_grade, store, known,
                        hist["prior_attach"], offers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shopper")
    ap.add_argument("--time", default="2026-06-15 07:30")
    ap.add_argument("--grade", default="REGULAR")
    ap.add_argument("--store", type=int)
    ap.add_argument("--demo", action="store_true",
                    help="pick a real shopper with history")
    args = ap.parse_args()

    eng = OfferEngine()
    shopper = args.shopper
    if args.demo or not shopper:
        busy = (eng.visit.group_by("shopper_id").len()
                .sort("len", descending=True).row(5)[0])
        shopper = busy
        print(f"(demo shopper {shopper})\n")

    d = eng.decide(shopper, args.time.replace(" ", "T"), args.grade, args.store)
    print(f"shopper   {d.shopper_id}{'' if d.known_shopper else '   [unseen]'}")
    print(f"when      {d.when:%Y-%m-%d %H:%M}  ({daypart_of(d.when)})")
    print(f"grade     {d.fuel_grade}        store {d.store_id}")
    pa = d.prior_attach
    print("prior     " + ("never seen" if pa < 0
                          else f"went inside {pa:.0%} of past fills") + "\n")

    if not d.ranked:
        print("no scorable candidates - check the catalog and the daypart")
        return
    print(f"category  {d.category}   (stage one: highest P we may show)")
    print("ranked candidates inside that category")
    for o in d.ranked:
        print("  " + str(o))
    print()
    best = d.show
    if best is None:
        print(f"DECISION: show nothing "
              f"(best EV ${d.ranked[0].expected_value:.4f} "
              f"< threshold ${C.MIN_EXPECTED_VALUE:.2f})")
    else:
        print(f"DECISION: show {best.brand} - {best.category}  "
              f"(EV ${best.expected_value:.4f})")


if __name__ == "__main__":
    main()
