# Pump-screen pipeline

A runnable version of the design in
[pipeline v2](https://byuibigdata.github.io/bdbook/slides/pipeline2.html):
give it a customer, a time and a fuel grade, and it tells you which offer to
put on the pump screen — or that you should show nothing.

Polars does the data work. Scikit-learn fits the two classifiers and
[`implicit`](https://github.com/benfred/implicit) fits the ALS recommender.

## Run it

```bash
cd data/pipeline_analysis
UV="uv run --with polars --with scikit-learn --with joblib --with implicit"

$UV python build_tables.py --stores 25   # a few minutes, for a dry run
$UV python train_models.py
$UV python validate.py                   # 50 checks; non-zero exit on failure
$UV python serve.py --demo
```

Drop `--stores 25` to build from all 940 stores and 128.6M line items.
That takes considerably longer and needs the memory to match.

## Ask it a question

```bash
$UV python serve.py \
    --shopper f91dcc0a80e3fd6c948d8bfe5ff32b61 \
    --time "2026-06-15 07:30" \
    --grade REGULAR
```

```
shopper   f91dcc0a80e3fd6c948d8bfe5ff32b61
when      2026-06-15 07:30  (morning)
grade     REGULAR        store 33577
P(comes inside) = 0.088

ranked candidates
  Red Bull   Packaged Beverages  EV $0.0020  P(in)=0.088 P(cat)=0.501 P(sku)=0.026  vendor-funded
  Monster    Packaged Beverages  EV $0.0012  P(in)=0.088 P(cat)=0.501 P(sku)=0.019  vendor-funded
  ...

DECISION: show Red Bull - Packaged Beverages (Non-Alcoholic)  (EV $0.0020)
```

Or as a library:

```python
from serve import OfferEngine

eng = OfferEngine()
d = eng.decide(shopper_id="f91dcc...", when="2026-06-15 07:30",
               fuel_grade="REGULAR")
print(d.show)        # the Offer to display, or None
```

An unknown shopper works too — they fall back to context plus overall
popularity, which is the honest answer rather than a failure.

## The files

| File | What it does |
|---|---|
| `config.py` | Every constant worth arguing about: the window, the dayparts, the categories, the assumed margin |
| `build_tables.py` | Bronze → `visit` (the label) → features → the SVD matrix → a stub offer catalog |
| `train_models.py` | Fits and calibrates the three models, writes `train_report.json` |
| `validate.py` | 50 checks across tables, leakage, models and serving |
| `negotiate.py` | Inverts the rule: the minimum share the brand must fund |
| `serve.py` | The decision, as a CLI and as a class |

## Train / test splitting

**Time, never random.** A random split lets a model learn from next month to
predict last month and report a score nobody can collect on.

| Split | Where | Used for |
|---|---|---|
| **Train** | `ts < TRAIN_CUTOFF` (2026-04-01) | fitting |
| ↳ calibration slice | last 15% of the shuffled train rows | the isotonic calibrator |
| ↳ internal validation | 10%, inside each GBT | early stopping |
| **Test** | `ts >= TRAIN_CUTOFF` | out-of-time metrics only |

`train_report.json` reports both: `auc` on the calibration slice, which is
out-of-sample but **in-period**, and `oot_auc` on the held-out period, which
is the number to quote. On the 25-store build they are 0.716 and 0.709 — a
0.007 drop, which is the shape you want.

**Model C is restricted to the training period too.** ALS feels like
preprocessing rather than a model, which is exactly why this gets forgotten;
factor all two years and the latent factors have already seen the purchases
they are being asked to predict. `validate.py` checks the interaction table
is strictly smaller than the eligible line count.

## The three models

| | Question | Estimator | Trained on |
|---|---|---|---|
| **A** | Will they come inside? | `HistGradientBoostingClassifier` + isotonic calibration | all fuel visits |
| **B** | Which category? | six binary GBTs | converting visits only |
| **C** | Which SKU? | `implicit.als.AlternatingLeastSquares` on shopper × sku@daypart | beverage purchases |

**A is calibrated, not just ranked.** Its output is multiplied by dollars, so
a well-ordered but badly-scaled probability is useless. `validate.py` checks
predicted mean against actual on the test period.

**B is six binary models, not one multiclass.** A basket holds several
categories at once, so a softmax would force them to compete and return
numbers that cannot be priced.

**C is real ALS** — alternating least squares for implicit feedback, Hu,
Koren & Volinsky (2008), via the `implicit` package. Scikit-learn has no ALS;
`TruncatedSVD` is the nearest thing and it optimises the wrong objective here,
treating an unobserved pair as a zero rather than as an absence of evidence.
ALS instead treats purchase counts as **confidence** — nobody scored a Red Bull
four out of five, they bought it nine times — and `alpha` controls how fast
that confidence grows.

The item key is `gtin@daypart`. ALS takes no features, so the only way to give
it a clock is to put one in the key, and the clock matters: same-daypart
purchase pairs repeat the same SKU 18.0% of the time against 10.0% across
dayparts.

## Things that are deliberately honest

**The offer catalog is a stub.** `build_tables.py` writes
`artifacts/offer_catalog.csv` with margins derived from average price times an
assumed 40%. **The PDI data has no cost column.** Edit that file with real
numbers and every decision changes.

`funded_share` is the share of the discount the **brand** pays, between 0 and
1 — not a flag. A brand often funds all of a promotion, and sometimes the
retailer and the brand split it:

    our_cost         = discount x (1 - funded_share)
    effective_margin = unit_margin - our_cost

At `funded_share = 1` the store's margin per unit is untouched and it simply
sells more volume, which is why these programmes exist.

**There is no `p₁`.** The decision rule should compare the world where we show
an offer against the world where we do not. Measuring that needs a holdout —
a slice of pumps that show nothing — and it does not exist. The engine
therefore prices every offer at `p₀`, which is why a self-funded discount can
never come out ahead here. That is a property of the missing experiment, not a
finding about discounts.

**Nothing is logged.** A production version would write a `decision_log` with
the offer shown *and the probability it was shown with*. Without logged
propensities no off-policy evaluation is possible later, and that information
cannot be reconstructed after the fact.

## Using it to negotiate

Read the decision rule backwards and it answers a commercial question: how
much of the discount does the brand need to cover before the offer earns a
slot on the screen?

```bash
$UV python negotiate.py --brand Monster --daypart morning --sweep
```

```
  brand           P(buy)  margin   now     +10%    +25%    +50%   +100%   +200%
  ------------------------------------------------------------------------------
  Red Bull        0.0007    1.50  100%      85%     67%     45%     17%     0%*
  Monster         0.0006    1.32  100%      87%     71%     51%     27%      2%
```

Writing the uplift as relative, `p1 = p0 (1 + r)`, the base rate **cancels**:

    f_min = 1 - (margin / discount) x r / (1 + r)

Worth saying out loud in the meeting: how likely the customer was to buy
decides *how much the deal is worth*, not *who should pay for it*. The floor
depends only on the margin-to-discount ratio and on how far the offer moves
people.

`r` is an assumption — without a holdout we cannot measure it — so the table
is the shape of the negotiation, not its answer.

## Validation

`validate.py` is the part to read first if you are reviewing this. The check
that matters most:

> a shopper's **first** visit must have `prior_n = 0`

If that fails, the history features are reading the outcome they are supposed
to predict. It caught a real bug during development — ties in `DATE_TIME` made
the "first visit" ordering non-deterministic, so 12 rows disagreed between the
feature builder and the check. Both now break ties on `fuel_tx`.

It also caught a second one worth knowing about: **isotonic calibration can
return exactly 0.** It is a step function fitted on finite data, so its lowest
step sits at zero when no positives landed in that bin — which is not a claim
that the event is impossible, but it would wipe out every expected value
downstream. Probabilities are floored at 1e-5 in `serve.py`.
