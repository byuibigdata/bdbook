"""Four customers, the same pump, four different offers.

Picks real shoppers who match an archetype -- never bought anything, a soda
regular, a cigarette regular, a salty-snack regular -- then runs all of them
through the identical context so the only thing that differs is who they are.

That isolates what the customer id actually buys us: Model A barely moves,
Model B moves a lot, Model C decides the SKU.

    uv run --with polars --with scikit-learn --with joblib --with implicit \
        python personas.py
    python personas.py --time "2026-06-15 17:45" --grade DIESEL
"""

from __future__ import annotations

import argparse
from datetime import datetime

import polars as pl

import config as C
from serve import OfferEngine, daypart_of

BEV = "Packaged Beverages (Non-Alcoholic)"
CIG = "Cigarettes"
SALT = "Salty Snacks"
BEER = "Beer"

PERSONAS = [
    ("never bought", None),
    ("soda regular", BEV),
    ("cigarette regular", CIG),
    ("salty snack regular", SALT),
    ("beer regular", BEER),
]


def shopper_profiles(eng: OfferEngine, cut: pl.Expr) -> pl.DataFrame:
    """Per shopper: fuel visits and the share containing each category."""
    v = eng.visit.filter(pl.col("ts") < cut)
    agg = [pl.len().alias("visits"), pl.col("bought").sum().alias("bought")]
    agg += [
        pl.col("cats").list.contains(c).fill_null(False).mean().alias(c)
        for c in (BEV, CIG, SALT, BEER)
    ]
    return v.group_by("shopper_id").agg(agg)


def pick(prof: pl.DataFrame, eng: OfferEngine, category: str | None) -> str | None:
    """A shopper who typifies the archetype and whom Model C can speak about."""
    in_als = pl.Series("s", list(eng.shopper_ix.keys())).implode()
    if category is None:
        # Fuels often, never once walked inside.
        cand = (prof.filter((pl.col("visits") >= 8) & (pl.col("bought") == 0))
                .sort(["visits", "shopper_id"], descending=[True, False]))
    else:
        others = [c for c in (BEV, CIG, SALT, BEER) if c != category]
        cand = (prof.filter(
                    (pl.col("visits") >= 8)
                    & (pl.col(category) >= 0.4)
                    & pl.col("shopper_id").is_in(in_als))
                # typical of the archetype, and not equally typical of another
                .with_columns(sep=pl.col(category) - pl.max_horizontal(others))
                .sort(["sep", "shopper_id"], descending=[True, False]))
    return None if cand.height == 0 else cand.row(0, named=True)["shopper_id"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", default="2026-06-15 07:12")
    ap.add_argument("--grade", default="REGULAR")
    args = ap.parse_args()

    eng = OfferEngine()
    when = datetime.fromisoformat(args.time.replace(" ", "T"))
    cut = pl.lit(C.TRAIN_CUTOFF).str.to_datetime()
    prof = shopper_profiles(eng, cut)

    print(f"\nSame pump, same moment, four different people")
    print(f"  {when:%Y-%m-%d %H:%M} · daypart {daypart_of(when)} · "
          f"grade {args.grade}\n")

    rows = []
    for label, cat in PERSONAS:
        sid = pick(prof, eng, cat)
        if sid is None:
            print(f"  [{label}] no match in this build")
            continue
        p = prof.filter(pl.col("shopper_id") == sid).row(0, named=True)
        d = eng.decide(sid, when, args.grade, top=10_000)
        pcat = eng.category_probs(sid, when, args.grade, store_id=d.store_id)
        best = d.show
        rows.append((label, sid, p, d, pcat, best))

    # ---- history ------------------------------------------------------
    print("  PRIOR BEHAVIOUR (before the cutoff)")
    print(f"  {'persona':<21}{'visits':>7}{'inside':>8}"
          f"{'soda':>8}{'cigs':>7}{'salty':>7}{'beer':>7}")
    for label, sid, p, *_ in rows:
        inside = p["bought"] / p["visits"] if p["visits"] else 0
        print(f"  {label:<21}{p['visits']:>7}{inside:>8.0%}"
              f"{p[BEV]:>8.0%}{p[CIG]:>7.0%}{p[SALT]:>7.0%}{p[BEER]:>7.0%}")

    # ---- category relevance -------------------------------------------
    print("\n  CATEGORY RELEVANCE (Binary Relevance) — P(buys it on this fill)")
    print(f"  {'persona':<21}{'soda':>9}{'cigs':>9}{'salty':>9}{'beer':>9}")
    for label, sid, p, d, pcat, _ in rows:
        print(f"  {label:<21}{pcat.get(BEV,0):>9.3f}{pcat.get(CIG,0):>9.3f}"
              f"{pcat.get(SALT,0):>9.3f}{pcat.get(BEER,0):>9.3f}")

    tops = [max(r[4].values()) for r in rows if r[4]]
    if tops:
        print(f"  {'':<21}spread on the leading category: "
              f"{max(tops) - min(tops):.4f}")

    # ---- decision -----------------------------------------------------
    print("\n  THE OFFER  (SKU Affinity + Offer Economics)")
    for label, sid, p, d, pcat, best in rows:
        if best is None:
            print(f"  {label:<21}show nothing")
            continue
        print(f"  {label:<21}{str(best.brand)[:16]:<17}"
              f"{best.category[:24]:<25}EV ${best.expected_value:.5f}")
    print()


if __name__ == "__main__":
    main()
