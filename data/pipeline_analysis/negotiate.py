"""What share of a discount does the brand have to fund?

The expected-value rule decides whether to show an offer. Read backwards, it
answers a commercial question instead: given what we predict this SKU will
do at this daypart, how much of the discount does the brand need to cover
before the offer is worth a slot on the screen?

    show iff   p1 (margin - discount x (1 - f))  >  p0 x margin

Write the uplift as a RELATIVE lift, p1 = p0 (1 + r) -- at SKU level that
is the only scale that means anything, since a single SKU's purchase
probability is a fraction of a percent and "plus half a point" would be an
eightfold lift. Solving for f, p0 cancels:

    f_min = 1 - (margin / discount) x r / (1 + r)

Which is worth saying out loud in the negotiation: **the floor does not
depend on how likely the customer was to buy.** It depends on the
margin-to-discount ratio and on how much the offer actually moves people.
How likely they were decides how much the deal is worth, not who should
pay for it.

At f = 1 the brand pays everything, the store's margin per unit is
untouched, and the only question left is whether the offer moves one extra
unit. At f = 0 the store funds it alone, and a $0.90 discount against a
$1.30 margin needs the purchase rate to more than triple.

Run:
    uv run --with polars --with scikit-learn --with joblib --with implicit \\
        python negotiate.py --brand Monster --daypart morning
    python negotiate.py --sweep        # the whole funding curve
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import polars as pl

import config as C
from serve import OfferEngine

# RELATIVE uplift: how much the offer multiplies the chance of buying this
# SKU. +50% means half again as likely. We have no holdout, so this is an
# input, and every number below is a function of it.
DEFAULT_UPLIFTS = [0.10, 0.25, 0.50, 1.00, 2.00]

DAYPART_HOUR = {"morning": 7, "midday": 12, "evening": 18}


def sample_shoppers(eng: OfferEngine, n: int, seed: int) -> list[str]:
    """Frequent shoppers, so the history features actually do something."""
    counts = (
        eng.visit.group_by("shopper_id").len()
        .filter(pl.col("len") >= 3)
        .sort("len", descending=True)
        .head(5000)
    )
    return counts.sample(min(n, counts.height), seed=seed)["shopper_id"].to_list()


def profile(eng: OfferEngine, brand: str | None, daypart: str,
            n_shoppers: int, grade: str, seed: int) -> pl.DataFrame:
    """Average the predicted purchase probability per SKU over real shoppers."""
    hour = DAYPART_HOUR[daypart]
    when = datetime(2026, 6, 15, hour, 30)
    rows: dict[str, list] = {}

    for sid in sample_shoppers(eng, n_shoppers, seed):
        d = eng.decide(sid, when, grade, top=10_000)
        for o in d.ranked:
            if brand and (o.brand or "") != brand:
                continue
            rows.setdefault(o.gtin, []).append(o)

    if not rows:
        return pl.DataFrame()

    recs = []
    for gtin, offers in rows.items():
        o = offers[0]
        recs.append({
            "gtin": gtin,
            "brand": o.brand,
            "p_buy": float(np.mean([x.p_buy for x in offers])),
            "unit_margin": o.unit_margin,
            "discount": o.discount,
            "funded_now": o.funded_share,
            "n": len(offers),
        })
    return pl.DataFrame(recs).sort("p_buy", descending=True)


def floor_for(margin: float, discount: float, rel_uplift: float) -> float:
    """Minimum brand funding share. p0 cancels, so it is not an argument."""
    if discount <= 0 or rel_uplift <= 0:
        return 1.0
    return 1.0 - (margin / discount) * (rel_uplift / (1.0 + rel_uplift))


def fmt(share: float) -> str:
    if share <= 0:
        return "   0%*"      # pays for itself
    if share > 1:
        return "  >100%"     # no amount of brand money fixes it
    return f"{share:6.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", help="limit to one brand, e.g. Monster")
    ap.add_argument("--daypart", default="morning",
                    choices=list(DAYPART_HOUR))
    ap.add_argument("--grade", default="REGULAR")
    ap.add_argument("--shoppers", type=int, default=40,
                    help="how many real shoppers to average over")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--sweep", action="store_true",
                    help="show the funding curve for the leading SKU")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    eng = OfferEngine()
    df = profile(eng, args.brand, args.daypart, args.shoppers, args.grade,
                 args.seed)
    if df.is_empty():
        raise SystemExit("no SKUs matched — check --brand and --daypart")

    print(f"\nMinimum share of the discount the BRAND must fund")
    print(f"  daypart {args.daypart} · grade {args.grade} · "
          f"averaged over {args.shoppers} shoppers"
          + (f" · brand {args.brand}" if args.brand else ""))
    print(f"  assumed RELATIVE uplift from showing the offer:\n")

    head = "  ".join(f"{u*100:>+5.0f}%" for u in DEFAULT_UPLIFTS)
    print(f"  {'brand':<14}{'P(buy)':>8}{'margin':>8}{'now':>6}   {head}")
    print("  " + "-" * (36 + 9 * len(DEFAULT_UPLIFTS)))

    for r in df.head(args.top).iter_rows(named=True):
        cells = "  ".join(
            fmt(floor_for(r["unit_margin"], r["discount"], u))
            for u in DEFAULT_UPLIFTS)
        print(f"  {str(r['brand'])[:13]:<14}{r['p_buy']:>8.4f}"
              f"{r['unit_margin']:>8.2f}{r['funded_now']:>6.0%}   {cells}")

    print("\n  * 0% means the offer pays for itself at that uplift.")
    print("    >100% means no amount of brand funding makes it worth running —")
    print("    the discount destroys more margin than the extra volume returns.")

    if args.sweep:
        r = df.row(0, named=True)
        print(f"\n  Funding curve for {r['brand']} "
              f"(margin ${r['unit_margin']:.2f}, discount ${r['discount']:.2f}, "
              f"P(buy) {r['p_buy']:.4f})\n")
        print(f"  {'brand funds':>12}{'our cost':>10}{'eff. margin':>13}"
              f"{'rel. uplift':>15}")
        for f in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
            cost = r["discount"] * (1 - f)
            eff = r["unit_margin"] - cost
            if eff <= 0:
                need = "never profitable"
            else:
                # p1 eff = p0 margin  ->  relative lift = margin/eff - 1
                need = f"{(r['unit_margin'] / eff - 1) * 100:+.0f}%"
            print(f"  {f:>11.0%}{cost:>10.2f}{eff:>13.2f}{need:>15}")

    print("\n  Every number here is conditional on the assumed uplift. Without")
    print("  a holdout we cannot measure it, so take these as the shape of the")
    print("  negotiation, not its answer.\n")


if __name__ == "__main__":
    main()
