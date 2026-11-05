"""The scoring core: one implementation, two callers.

`serve.py` answers for one customer standing at a pump. `score_table.py`
writes a row for every customer-moment in the test period. They used to each
implement the ranking, and the two copies drifted twice -- once the batch
path lost the within-category normalisation, once it lost P(category) out of
the expected value. Both times the symptom was the same: a beer SKU in the
top slot for a customer whose beer propensity was 0.004, because beer carries
the fattest margin in the catalog.

So the ranking lives here, written for N rows. Serving is the N = 1 case,
which means the cheap path and the expensive path cannot disagree.

The decision itself is two stages: relevance picks the category, economics
picks the SKU inside it. See `SkuRanker`.

Everything here is pure: models and tables come in as arguments.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

# Isotonic regression is a step function fitted on finite data, so its lowest
# step can sit at exactly 0 -- which means "no positives landed in this bin",
# not "this cannot happen". A hard zero wipes out every expected value
# downstream, and a hard 1 makes prob_entering exactly 1.0, which is not a
# thing we know. Floor and cap both ends.
P_FLOOR = 1e-5

# Rows per block when scoring SKUs. The score matrix is rows x items, so a
# full test period against a few hundred items per daypart would otherwise
# allocate gigabytes in one go.
CHUNK = 40_000


def slug(name: str) -> str:
    """Category name -> column-safe token. Must match train_models.slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------- features
def design_matrix(df: pl.DataFrame, cat_maps: dict, cols: list[str]
                  ) -> np.ndarray:
    """Encode categoricals the way training did and lay out the columns.

    Unseen levels become -1, which HistGradientBoosting treats as its own
    category. The column ORDER is the contract with the fitted model, so it
    comes from the saved `cols` list rather than from the frame.
    """
    enc = {c: df[c].fill_null("__NA__").cast(pl.Utf8)
           .replace_strict(m, default=-1, return_dtype=pl.Int32)
           for c, m in cat_maps.items()}
    return df.with_columns(**enc).select(cols).fill_null(0).to_numpy()


# ------------------------------------------------- Category Relevance (GBT)
def category_probs(models: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """P(buys this category on this fill), per category, one array each.

    Unconditional. There is no separate "will they come inside" step: coming
    inside is already inside these numbers.
    """
    return {cat: np.clip(m.predict_proba(X)[:, 1], P_FLOOR, 1.0 - P_FLOOR)
            for cat, m in models.items()}


def prob_entering(props: dict[str, np.ndarray]) -> np.ndarray:
    """P(enters the store) recovered from the six categories.

    Under independence, P(none of them) = prod(1 - p_i). Against a dedicated
    attach model scoring AUC 0.827 this reaches 0.822 and sits within a point
    on the mean, which is why the dedicated model was dropped.

    Report it. Do NOT multiply it back into an expected value -- the category
    term is already in there, and counting it twice shrinks every offer.
    """
    P = np.column_stack([props[k] for k in sorted(props)])
    return 1.0 - np.prod(1.0 - P, axis=1)


# ----------------------------------------------------- SKU Affinity (ALS)
class SkuRanker:
    """Picks a category, then ranks that category's SKUs, for a batch of fills.

    The decision is two stages on purpose:

      1. **Category, by relevance.** The most likely category we are allowed
         to show. P(category) only -- no margin.
      2. **SKU, by economics.** Inside that one category, expected value.

    Nothing from another category can reach the screen, however fat its
    margin. An earlier single-stage version ranked every SKU in the catalog
    against every other by expected value, and offered candy to a customer we
    were 90.6% sure wanted a beverage: `p_sku` is normalised within a
    category, so a four-SKU category concentrates its probability where a
    three-hundred-SKU category spreads it thin, and a 2% category with a
    short tail out-earned a 90% category with a long one. Those expected
    values were not comparable, because their denominators were not.

    ALS gives a factor vector per shopper and per item; affinity is their dot
    product. Items are keyed `gtin@daypart`, so restricting to one daypart is
    a filter on the item side -- which is how a model that takes no features
    ends up knowing it is 6am. A shopper with no factor vector falls back to
    popularity, which is roughly three fills in four and is an honest answer
    rather than a failure.
    """

    def __init__(self, aff: dict, catalog: pl.DataFrame) -> None:
        self.user_factors = np.asarray(aff["als"].user_factors)
        self.item_factors = np.asarray(aff["als"].item_factors)
        self.shopper_ix = aff["shopper_ix"]
        self.items = aff["items"]
        self.popularity = np.asarray(aff["popularity"], dtype=float)

        # The FULL catalog, deliberately. P(sku | category) is a conditional
        # over every SKU in that category; stock-outs and the tobacco rule
        # restrict what we may SHOW, and letting them shrink the denominator
        # would quietly inflate whatever is left.
        self.meta = {r["gtin"]: r for r in catalog.iter_rows(named=True)}
        cat_of = {r["gtin"]: r["category"] for r in catalog.iter_rows(named=True)}

        # daypart -> category -> item indices
        self.buckets: dict[str, dict[str, list[int]]] = {}
        for i, it in enumerate(self.items):
            gtin, _, dp = it.rpartition("@")
            cat = cat_of.get(gtin)
            if cat is not None:
                self.buckets.setdefault(dp, {}).setdefault(cat, []).append(i)

        # Units sold per category per daypart, over the SKUs we may show.
        # Used only to break ties in stage one, which matters more than it
        # sounds: a brand-new customer sits at the probability floor in every
        # category, so without this the winner is decided by the alphabet,
        # and "Beer" wins. Falling back to the daypart's best-selling
        # category is the same Best-Seller logic stage two already uses.
        self.cat_popularity: dict[str, dict[str, float]] = {}
        for dp, bycat in self.buckets.items():
            self.cat_popularity[dp] = {
                cat: float(sum(
                    self.popularity[i] for i in members
                    if self.meta[self.items[i].rpartition("@")[0]]["in_stock"]
                    and self.meta[self.items[i].rpartition("@")[0]]["offerable"]))
                for cat, members in bycat.items()}

    def personalised(self, shoppers: list[str]) -> np.ndarray:
        """Whether ALS has a factor vector for each shopper.

        Worth carrying next to the scores, because it changes what the raw
        affinity number MEANS: a dot product for the personalised path, a
        unit count for the popularity fallback. `p_sku` is comparable across
        both; `affinity` is not.
        """
        return np.array([s in self.shopper_ix for s in shoppers])

    def _raw(self, shoppers: list[str], idx: np.ndarray) -> np.ndarray:
        """ALS affinity, or popularity for shoppers we have no vector for."""
        known = np.array([self.shopper_ix.get(s, -1) for s in shoppers])
        out = np.empty((len(shoppers), idx.size), dtype=float)
        hit = known >= 0
        if hit.any():
            out[hit] = self.user_factors[known[hit]] @ self.item_factors[idx].T
        if (~hit).any():
            out[~hit] = self.popularity[idx]
        return np.clip(out, 0, None)

    def _availability(self, dp: str, props: dict[str, np.ndarray]
                      ) -> dict[str, np.ndarray]:
        """Categories we could actually run an offer in, for this daypart.

        A category qualifies when we have a relevance model for it and at
        least one of its SKUs is in stock and allowed on the screen. The
        tobacco rule knocks out whole categories here, which is why the
        customer we are 87% sure wants cigarettes gets a drink instead.
        """
        out = {}
        for cat, members in (self.buckets.get(dp) or {}).items():
            if cat not in props:
                continue          # no P(category) means no expected value
            show = any(
                self.meta[self.items[i].rpartition("@")[0]]["in_stock"]
                and self.meta[self.items[i].rpartition("@")[0]]["offerable"]
                for i in members)
            if show:
                out[cat] = np.asarray(members)
        return out

    def choose_category(self, dayparts: list[str],
                        props: dict[str, np.ndarray]) -> list[str | None]:
        """Stage one: the category to run the offer in, per row.

        The most relevant category we are allowed to show, by P(category),
        with ties going to the daypart's best-selling category.
        Deliberately NOT by expected value. Letting margin pick the category
        offers candy to a customer we are 90% sure wants a soda, because
        `p_sku` is normalised within a category and a four-SKU category
        concentrates its probability where a three-hundred-SKU category
        spreads it thin. Comparing expected values across categories of
        different width is comparing numbers with different denominators.

        So the split is: **relevance picks the category, economics picks the
        SKU inside it.** Each stage answers the question it can actually
        answer.
        """
        n = len(dayparts)
        pick: list[str | None] = [None] * n
        for dp in sorted(set(dayparts)):
            avail = self._availability(dp, props)
            if not avail:
                continue
            # Most popular first, so np.argmax's first-index tie-break lands
            # on the daypart's best seller rather than on the alphabet.
            pop = self.cat_popularity.get(dp, {})
            cats = sorted(avail, key=lambda c: (-pop.get(c, 0.0), c))
            rows = np.array([k for k, d in enumerate(dayparts) if d == dp])
            P = np.column_stack([props[c][rows] for c in cats])
            won = np.asarray(cats)[np.argmax(P, axis=1)]
            for k, c in zip(rows, won):
                pick[k] = str(c)
        return pick

    def rank(self, shoppers: list[str], dayparts: list[str],
             props: dict[str, np.ndarray], top: int = 10
             ) -> list[list[dict]]:
        """Top-`top` offers per row, best expected value first.

        Two stages. `choose_category` picks one category per row; the SKUs
        returned here all come from that one category, ranked by `p_buys`.
        Nothing from another category can appear.

        This answers "what will they buy anyway", on probability alone. The
        margin fields ride along for `negotiate.py`, which is explicit about
        resting on an assumed margin, but they do not set the order.

        `props` is keyed by category NAME, each value an array as long as
        `shoppers`. Returns one list of candidate dicts per input row.
        """
        n = len(shoppers)
        if not (n == len(dayparts) and all(len(v) == n for v in props.values())):
            raise ValueError("shoppers, dayparts and props must agree in length")
        out: list[list[dict]] = [[] for _ in range(n)]
        chosen = self.choose_category(dayparts, props)

        # Group rows by (daypart, chosen category) so one matrix product
        # serves every row that landed in the same bucket.
        groups: dict[tuple[str, str], list[int]] = {}
        for k, (dp, cat) in enumerate(zip(dayparts, chosen)):
            if cat is not None:
                groups.setdefault((dp, cat), []).append(k)

        for (dp, cat), rows_all in groups.items():
            idx = self._availability(dp, props)[cat]
            gtins = [self.items[i].rpartition("@")[0] for i in idx]
            # Showable restricts the CHOICE, not the probability: p_sku is
            # still normalised over every SKU in the category, so a stock-out
            # does not quietly inflate what is left on the shelf.
            showable = np.array(
                [bool(self.meta[g]["in_stock"]) and bool(self.meta[g]["offerable"])
                 for g in gtins])
            margin = np.array([float(self.meta[g]["unit_margin"]) for g in gtins])
            cost = np.array([float(self.meta[g]["our_cost"]) for g in gtins])
            k_top = min(top, int(showable.sum()))

            for s in range(0, len(rows_all), CHUNK):
                rows = rows_all[s:s + CHUNK]
                raw = self._raw([shoppers[k] for k in rows], idx)
                tot = raw.sum(axis=1, keepdims=True)
                # P(this SKU | this category).
                p_sku = np.divide(raw, tot, out=np.zeros_like(raw), where=tot > 0)
                p_cat = props[cat][rows][:, None]

                # P(buys this exact SKU on this fill) -- the two stages
                # multiplied together. Inside one category P(category) is a
                # constant, so this orders the same way p_sku does.
                p_buys = p_cat * p_sku
                ev = p_buys * (margin - cost)

                # Ranked on probability, NOT on expected value. The catalog's
                # margins are average price times an assumed 40% -- the PDI
                # data has no cost column -- so ranking on dollars ranks on
                # an assumption. These are the items they are most likely to
                # buy, which is a claim the data supports.
                ranked_on = np.where(showable, p_buys, -np.inf)
                order = np.argsort(-ranked_on, axis=1, kind="stable")[:, :k_top]

                for r, k in enumerate(rows):
                    picks = []
                    for rank_, j in enumerate(order[r], start=1):
                        if not np.isfinite(ranked_on[r, j]):
                            break
                        g = gtins[j]
                        m = self.meta[g]
                        picks.append({
                            "rank": rank_,
                            "gtin": g,
                            "brand": m["brand"],
                            "category": m["category"],
                            "affinity": float(raw[r, j]),
                            "p_category": float(p_cat[r, 0]),
                            "p_sku": float(p_sku[r, j]),
                            "p_buys": float(p_buys[r, j]),
                            "unit_margin": float(m["unit_margin"]),
                            "discount": float(m["discount"]),
                            "funded_share": float(m["funded_share"]),
                            "our_cost": float(m["our_cost"]),
                            "expected_value": float(ev[r, j]),
                        })
                    out[k] = picks
        return out
