"""Walk one arriving customer through the whole pipeline, step by step.

`serve.py` returns an answer. This prints the reasoning: what we know the
moment the card is read, what we look up, what each model says, and how the
three probabilities become one offer priced in dollars.

    uv run --with polars --with scikit-learn --with joblib --with implicit \\
        python explain.py --demo
    python explain.py --shopper <id> --time "2026-06-15 07:12" --grade REGULAR

Add --json to emit the same trace as a dict, which is what the write-up
page is generated from.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import polars as pl

import config as C
from serve import OfferEngine, daypart_of

RULE = "─" * 74


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n {n}. {title}\n{RULE}")


def pick_demo(eng: OfferEngine) -> str:
    """A shopper worth walking through: one the models can speak about.

    Needs real fuel history, a record of actually going inside, and a row in
    the ALS matrix -- otherwise the trace shows three fallbacks rather than
    three models.
    """
    have_history = pl.Series("shopper_id", list(eng.shopper_ix.keys()))
    counts = (
        eng.visit.group_by("shopper_id")
        .agg(visits=pl.len(), bought=pl.col("bought").sum())
        .filter(pl.col("shopper_id").is_in(have_history.implode()))
        .filter(pl.col("visits").is_between(6, 60) & (pl.col("bought") >= 3))
        # shopper_id breaks ties so --demo picks the same person twice.
        .sort(["bought", "shopper_id"], descending=[True, False])
    )
    if counts.height == 0:                       # nobody qualifies; take any
        return eng.visit["shopper_id"][0]
    return counts.row(min(3, counts.height - 1))[0]


def trace(eng: OfferEngine, shopper: str, when: datetime, grade: str,
          gallons_dollars: float | None, quiet: bool = False) -> dict:
    out: dict = {}
    dp = daypart_of(when)
    store = eng._home_store(shopper)
    known_store = store is not None
    if store is None:
        store = int(eng.visit.group_by("store_id").len()
                    .sort("len", descending=True).row(0)[0])
    spend = gallons_dollars if gallons_dollars is not None else eng.default_spend

    # ---------------------------------------------------------------- 1
    if not quiet:
        head(1, "A car pulls up. What the pump knows before any lookup")
        print(f"   card read          {shopper[:12]}…   (hash of BIN + last 4 + state)")
        print(f"   clock              {when:%Y-%m-%d %H:%M}  ->  daypart {dp}")
        print(f"   grade selected     {grade}")
        print(f"   dollars pumped     ${spend:,.2f}   (live, mid-fill)")
        print(f"   store              {store}"
              f"{'' if known_store else '   [inferred: busiest store]'}")
        print("\n   None of this needs a database. The card is read before the")
        print("   fuel flows, so identity is available at the start of the fill.")
    out["inputs"] = {"shopper": shopper, "when": when.isoformat(), "daypart": dp,
                     "grade": grade, "fuel_spend": round(spend, 2),
                     "store": store, "store_known": known_store}

    # ---------------------------------------------------------------- 2
    hist = eng._history(shopper, when)
    prior = eng.visit.filter(
        (pl.col("shopper_id") == shopper) & (pl.col("ts") < when))
    n_prior = prior.height
    n_bought = int(prior["bought"].sum()) if n_prior else 0
    if not quiet:
        head(2, "One lookup: what has this customer done before?")
        print(f"   prior fuel visits  {n_prior}")
        print(f"   of those, bought   {n_bought}"
              + (f"   ({n_bought / n_prior:.0%})" if n_prior else ""))
        print(f"\n   {'category':<36}{'bought before':>14}{'share of visits':>17}")
        for cat in C.TARGET_CATEGORIES:
            print(f"   {cat[:35]:<36}"
                  f"{'yes' if hist[f'bought_before_{cat}'] else 'no':>14}"
                  f"{hist[f'prior_share_{cat}']:>17.1%}")
        print("\n   Computed strictly from visits BEFORE this one. In production")
        print("   this is a nightly table keyed on the shopper, not a live scan.")
    out["history"] = {
        "prior_visits": n_prior, "prior_bought": n_bought,
        "by_category": {c: {"bought_before": int(hist[f"bought_before_{c}"]),
                            "prior_share": round(hist[f"prior_share_{c}"], 4)}
                        for c in C.TARGET_CATEGORIES}}

    d = eng.decide(shopper, when, grade, store_id=store, fuel_spend=spend,
                   top=10_000)

    # ---------------------------------------------------------------- 3
    cats = eng.category_probs(shopper, when, grade, store_id=store,
                              fuel_spend=spend)
    if not quiet:
        head(3, "Category Relevance (Binary Relevance) — what will they buy?")
        print("   Six calibrated gradient-boosted classifiers, one per")
        print("   category, each scoring THIS fill directly.\n")
        for cat, p in sorted(cats.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(p * 300)
            mark = "" if cat not in C.NOT_OFFERABLE else "   [not offerable]"
            print(f"   {cat[:34]:<36}{p:7.4f}  {bar}{mark}")
        pa = hist["prior_attach"]
        print()
        if pa < 0:
            print("   This shopper is new to us, so only context is driving these.")
        else:
            print(f"   Driven by context plus this shopper's own record: they went")
            print(f"   inside on {pa:.0%} of {hist['prior_n']} prior fills, and their")
            print("   category mix is in the features too.")
        print("\n   There is no separate 'will they come inside' model. These")
        print("   probabilities are unconditional — coming inside is already")
        print("   inside them. Combined under independence they put this")
        print(f"   customer's chance of walking in at "
              f"{1 - np.prod([1 - p for p in cats.values()]):.1%}.")
    out["category_relevance"] = {k: round(v, 5) for k, v in cats.items()}
    out["prob_entering"] = round(
        float(1 - np.prod([1 - p for p in cats.values()])), 5)

    # ---------------------------------------------------------------- 5
    personal = shopper in eng.shopper_ix
    if not quiet:
        head(4, "SKU Affinity (ALS) — which item inside that category?")
        print(f"   Stage one chose: {d.category}")
        print("   Stage two may only look inside it. A fatter margin in a")
        print("   different category cannot reach the screen — relevance picks")
        print("   the category, economics picks the item.\n")
        print(f"   ALS factor vector for this shopper: "
              f"{'yes' if personal else 'NO — falling back to popularity'}")
        print("   Ranked by expected value, not by P(sku) — inside one category"
              "\n   a thin margin on a likely item can lose to a fat margin on"
              "\n   a less likely one.\n")
        print(f"   {'brand':<16}{'P(sku)':>9}{'margin':>9}{'brand funds':>13}"
              f"{'our cost':>10}")
        for o in d.ranked[:5]:
            print(f"   {str(o.brand)[:15]:<16}{o.p_sku:>9.4f}{o.unit_margin:>9.2f}"
                  f"{o.funded_share:>13.0%}{o.our_cost:>10.2f}")
        if not personal:
            print("\n   This is the 81% case. The ranking is the store's best")
            print("   sellers for this daypart — a real offer, just not a personal one.")
    out["sku_affinity"] = {"personalised": personal, "category": d.category,
                      "top": [{"brand": o.brand, "p_sku": round(o.p_sku, 5),
                               "margin": o.unit_margin,
                               "funded_share": o.funded_share,
                               "our_cost": round(o.our_cost, 3)}
                              for o in d.ranked[:5]]}

    # ---------------------------------------------------------------- 6
    best = d.show
    if not quiet:
        head(5, "Offer Economics — two probabilities become one number")
        o = d.ranked[0]
        print(f"   P(category)  {o.p_category:.4f}")
        print(f" × P(sku)       {o.p_sku:.4f}")
        print(f"   {'':=<34}")
        print(f"   P(buys this) {o.p_buy:.6f}")
        print(f" × effective margin  ${o.unit_margin:.2f} − ${o.our_cost:.2f}"
              f" = ${o.unit_margin - o.our_cost:.2f}")
        print(f"   expected value     ${o.expected_value:.5f} per fill\n")
        if best is None:
            print("   DECISION: show nothing. Every candidate loses money.")
        else:
            print(f"   DECISION: {best.brand} — {best.category}")
        print(f"\n   Negotiating floor: for this SKU the brand must fund at least")
        for r in (0.25, 0.50, 1.00):
            f = o.min_funded_share(r)
            # A negative floor means the offer already pays for itself; the
            # brand owes us nothing, so clamp rather than print nonsense.
            shown = "none" if f <= 0 else f"{f:.0%}"
            tail = "  (pays for itself)" if f <= 0 else ""
            print(f"     {shown:>5}  if the offer lifts purchase by {r:+.0%}{tail}")
        print("\n   No holdout exists, so that lift is an assumption and this")
        print("   expected value is priced at p0 for both worlds.")
    out["decision"] = {
        "shown": None if best is None else
                 {"brand": best.brand, "category": best.category,
                  "expected_value": round(best.expected_value, 6)},
        "floors": {f"{int(r*100)}%": round(d.ranked[0].min_funded_share(r), 4)
                   for r in (0.25, 0.50, 1.00)} if d.ranked else {}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shopper")
    ap.add_argument("--time", default="2026-06-15 07:12")
    ap.add_argument("--grade", default="REGULAR")
    ap.add_argument("--spend", type=float, help="dollars pumped so far")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit the trace as JSON")
    args = ap.parse_args()

    eng = OfferEngine()
    shopper = args.shopper or pick_demo(eng)
    when = datetime.fromisoformat(args.time.replace(" ", "T"))

    out = trace(eng, shopper, when, args.grade, args.spend, quiet=args.json)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"\n{RULE}\n")


if __name__ == "__main__":
    main()
