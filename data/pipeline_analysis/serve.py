"""Answer the pump-screen question for one customer.

Give it a shopper, a time and a fuel grade; it returns the offer to show, or
tells you to show nothing.

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


def daypart_of(ts: datetime) -> str:
    return C.daypart_of(ts.hour)


@dataclass
class Offer:
    gtin: str
    brand: str
    category: str
    p_inside: float
    p_category: float
    p_sku: float
    unit_margin: float
    discount: float
    funded_share: float      # share of the discount the BRAND pays
    our_cost: float          # discount x (1 - funded_share)
    expected_value: float

    @property
    def p_buy(self) -> float:
        return self.p_inside * self.p_category * self.p_sku

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
                f"P(in)={self.p_inside:.3f} P(cat)={self.p_category:.3f} "
                f"P(sku)={self.p_sku:.3f}  {funder}")


@dataclass
class Decision:
    shopper_id: str
    when: datetime
    fuel_grade: str
    store_id: int
    known_shopper: bool
    p_inside: float
    ranked: list[Offer] = field(default_factory=list)

    @property
    def show(self) -> Offer | None:
        """The offer worth showing, or None. Showing nothing is a real answer."""
        if self.ranked and self.ranked[0].expected_value >= C.MIN_EXPECTED_VALUE:
            return self.ranked[0]
        return None


class OfferEngine:
    """Loads the artifacts once and scores a fill in milliseconds."""

    def __init__(self) -> None:
        a = joblib.load(C.ART / "model_a.joblib")
        b = joblib.load(C.ART / "model_b.joblib")
        c = joblib.load(C.ART / "model_c.joblib")
        self.model_a, self.cat_maps, self.cols_a = a["model"], a["cat_maps"], a["cols"]
        self.models_b, self.cols_b = b["models"], b["cols"]
        self.hist_cols = b["hist_cols"]
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
        # Median fill, used when the caller does not supply one.
        self.default_spend = float(self.visit["fuel_spend"].median())

    # -- features ---------------------------------------------------------
    def _encode(self, row: dict) -> dict:
        out = dict(row)
        for col, mapping in self.cat_maps.items():
            out[col] = mapping.get(str(row.get(col, "__NA__")), -1)
        return out

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
        for cat in C.TARGET_CATEGORIES:
            hits = (prior["cats"].list.contains(cat).fill_null(False).sum()
                    if n else 0)
            out[f"prior_n_{cat}"] = n
            out[f"bought_before_{cat}"] = int(hits > 0)
            out[f"prior_share_{cat}"] = (hits / n) if n else 0.0
        return out

    # -- the three models -------------------------------------------------
    # Isotonic regression is a step function fitted on finite data, so its
    # lowest step can sit at exactly 0 -- it means "no positives landed in
    # this bin", not "this cannot happen". A hard zero would wipe out every
    # expected value downstream, so floor it.
    P_FLOOR = 1e-5

    def _p_inside(self, ctx: dict) -> float:
        X = np.array([[ctx[c] for c in self.cols_a]], dtype=float)
        p = float(self.model_a.predict_proba(X)[0, 1])
        return min(max(p, self.P_FLOOR), 1.0 - self.P_FLOOR)

    def _p_categories(self, ctx: dict, hist: dict) -> dict[str, float]:
        row = {**ctx, **hist}
        X = np.array([[row.get(c, 0) for c in self.cols_b]], dtype=float)
        return {cat: min(max(float(m.predict_proba(X)[0, 1]), self.P_FLOOR),
                         1.0 - self.P_FLOOR)
                for cat, m in self.models_b.items()}

    def _sku_scores(self, shopper_id: str, daypart: str) -> dict[str, float]:
        """Score this daypart's items for the shopper, using the ALS factors.

        ALS gives a factor vector per shopper and per item; the affinity is
        their dot product. Items are keyed `gtin@daypart`, so restricting to
        one daypart is a filter on the item side -- which is how a model that
        takes no features ends up knowing it is 6am.

        An unseen shopper has no factor vector, so they fall back to overall
        popularity. That is the honest answer, not a failure.
        """
        suffix = "@" + daypart
        idx = np.array([i for i, it in enumerate(self.items)
                        if it.endswith(suffix)], dtype=np.int64)
        if idx.size == 0:
            return {}
        ix = self.shopper_ix.get(shopper_id)
        if ix is None:
            raw = self.popularity[idx]
        else:
            user_vec = self.als.user_factors[ix]
            raw = self.als.item_factors[idx] @ user_vec
        raw = np.clip(np.asarray(raw, dtype=float), 0, None)
        total = raw.sum()
        if total <= 0:
            return {}
        return {self.items[i].split("@")[0]: float(v / total)
                for i, v in zip(idx, raw)}

    # -- the decision -----------------------------------------------------
    def decide(self, shopper_id: str, when: str | datetime,
               fuel_grade: str, store_id: int | None = None,
               fuel_spend: float | None = None, top: int = 5) -> Decision:
        ts = when if isinstance(when, datetime) else datetime.fromisoformat(when)
        store = store_id if store_id is not None else self._home_store(shopper_id)
        known = store is not None
        if store is None:                       # never seen: use the busiest store
            store = int(self.visit.group_by("store_id").len()
                        .sort("len", descending=True).row(0)[0])

        ctx = self._encode({
            "daypart": daypart_of(ts), "fuel_grade": fuel_grade,
            "store_id": str(store), "hour": ts.hour,
            "weekday": ts.isoweekday(), "month": ts.month,
            "fuel_spend": fuel_spend if fuel_spend is not None else self.default_spend,
        })
        p_in = self._p_inside(ctx)
        p_cat = self._p_categories(ctx, self._history(shopper_id, ts))
        p_sku = self._sku_scores(shopper_id, daypart_of(ts))

        offers: list[Offer] = []
        for r in self.catalog.iter_rows(named=True):
            if not r["in_stock"]:
                continue
            pc = p_cat.get(r["category"])
            ps = p_sku.get(r["gtin"])
            if pc is None or ps is None:
                continue
            funded = float(r["funded_share"])
            discount = float(r["discount"])
            our_cost = discount * (1.0 - funded)
            margin = float(r["unit_margin"])
            # Offer vs no offer. Without a holdout we have no measured p1,
            # so this uses p0 for both -- which is exactly why a self-funded
            # discount can never come out ahead here. See the holdout note.
            p_buy = p_in * pc * ps
            ev = p_buy * (margin - our_cost)
            offers.append(Offer(r["gtin"], r["brand"], r["category"], p_in, pc,
                                ps, margin, discount, funded, our_cost, ev))

        offers.sort(key=lambda o: o.expected_value, reverse=True)
        return Decision(shopper_id, ts, fuel_grade, store, known, p_in, offers[:top])


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
    print(f"P(comes inside) = {d.p_inside:.3f}\n")

    if not d.ranked:
        print("no scorable candidates - check the catalog and the daypart")
        return
    print("ranked candidates")
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
