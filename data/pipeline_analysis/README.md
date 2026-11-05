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

$UV python build_tables.py --pilot       # the 24 pinned pilot stores
$UV python train_models.py
$UV python pairing.py --build            # the complement table
$UV python validate.py                   # 116 checks; non-zero exit on failure
$UV python validate_sku.py               # does stage two name the right SKU?
$UV python serve.py --demo               # what they buy anyway
$UV python pairing.py --demo             # what to discount instead
$UV python score_table.py --limit 0      # the whole held-out period to parquet
```

`--pilot` builds from the 24 stores pinned in `config.PILOT_STORES`, and
**every model figure in these write-ups comes from those stores.** They were
originally selected by `head(25)` over whatever order the parquet scan
returned, which is not reproducible — a later rebuild silently drew a
different 25 and moved every published number. Pinning the list fixes that.

Dropping the flag builds all 708 stores and 109.7M sellable lines. The
exploration figures quoted here (36.1M fuel visits, 12.8% attach) *are*
full-estate; only the fitted models are the pilot subset, and the two should
never be mixed. Note that `train_models.py` **cannot** fit the full estate as
written: `store_id` is a GBT categorical and HistGradientBoosting caps those
at 255 levels, so 708 stores raises a `ValueError`. Hashing or target-encoding
the store would be the fix.

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
| `train_models.py` | Fits and calibrates both models, writes `train_report.json` |
| `validate_sku.py` | Stage two on the held-out period: hit@k for ALS against the best seller, **on the same fills** |
| `pairing.py` | The **second** item: basket co-occurrence, cohort complementarity, and the pair offer |
| `scoring.py` | **The scoring core.** Encoding, category probabilities, `prob_entering`, and the expected-value ranking. Written for N rows |
| `serve.py` | The decision for one arriving customer — the N = 1 case of `scoring.py` — as a CLI and as a class |
| `score_table.py` | The same scoring over every test-period fill, written to `customer_scores.parquet` |
| `validate.py` | 116 checks across tables, leakage, models, serving, and batch/live agreement |
| `negotiate.py` | Inverts the rule: the minimum share the brand must fund |
| `explain.py` | One customer, traced step by step |
| `personas.py` | Five archetype customers through identical context |

### Why `scoring.py` exists

`serve.py` and `score_table.py` each used to implement the ranking. The two
copies drifted twice. Once the batch path lost the within-category
normalisation of `p_sku`; once it dropped `P(category)` out of the expected
value. Both times the symptom was identical — every top slot in the table was
beer, because beer carries the fattest margin in the catalog — and both times
it was caught by reading the output, not by a test.

So the ranking lives in one batch-shaped function and serving calls it with
one row. `validate.py` scores twelve real fills both ways and requires the
category probabilities, the expected values and the chosen SKU to match
exactly:

```
[ok] live and batch agree on P(category)    largest gap 0.00e+00
[ok] live and batch agree on expected value largest gap 0.00e+00
[ok] live and batch pick the same SKU       12/12 fills
```

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

**SKU Affinity is restricted to the training period too.** ALS feels like
preprocessing rather than a model, which is exactly why this gets forgotten;
factor all two years and the latent factors have already seen the purchases
they are being asked to predict. `validate.py` checks the interaction table
is strictly smaller than the eligible line count.

## The decision is two stages

```
stage 1   CATEGORY, by relevance     argmax P(category) over categories we may show
stage 2   SKU, by economics          argmax p_sku x effective margin, INSIDE that category
```

**Stage two may not leave the category stage one chose.** This matters, and
it was wrong at first. A single-stage version ranked every SKU in the catalog
against every other by expected value, and offered *candy* to a customer the
model was 90.6% sure wanted a beverage. The reason is not a bug but a
denominator: `p_sku` is normalised within its category, so a four-SKU
category concentrates its probability where a three-hundred-SKU category
spreads it thin. A 2% category with a short tail out-earns a 90% category
with a long one. Those expected values were never comparable.

Splitting it gives each stage the question it can actually answer. Relevance
decides *what the customer wants*; economics decides *which version of that
thing to put on the screen*. Margin can no longer overrule relevance.

Ties in stage one go to the daypart's **best-selling allowed category**. A
brand-new customer sits at the probability floor in all six, and without that
rule the winner is decided by the alphabet — which handed beer to the
customer we know nothing about.

## The two models

| Name | Question | Method | Trained on |
|---|---|---|---|
| **Category Relevance** | Will they buy category X on this fill? | Binary relevance — six calibrated `HistGradientBoostingClassifier` | every fuel visit |
| **SKU Affinity** | Which item inside that category? | ALS matrix factorisation (`implicit`) | purchases before the cutoff |

Two supporting pieces that are not models:

| Name | What it does |
|---|---|
| **Best-Seller Fallback** | Popularity ranking, for the ~76% with no category history |
| **Offer Economics** | Expected value under a funding share, and the call to show nothing. Stage two only |

### There used to be a third

An **attach model** answered "will they come inside at all", and its output
was multiplied into the others. It was dropped after being measured:
`P(category) = P(inside) x P(category | inside)` is an identity, so two
stages and one direct model scored the same to three decimals. It cost an
extra moving part and bought no accuracy.

What we gave up is an explicit `P(inside)` to report. If that number is
wanted back it is a cheap seventh binary model on the same features — for
dashboards, kept out of the expected-value chain so nothing double-counts.

**Category Relevance is calibrated, not just ranked.** Its output is
multiplied by dollars, so a well-ordered but badly-scaled probability is
useless. `validate.py` checks predicted mean against actual per category on
the test period.

**It is six binary models, not one multiclass.** A basket holds several
categories at once, so a softmax would force them to compete and return
numbers that cannot be priced.

**SKU Affinity is real ALS** — alternating least squares for implicit
feedback, Hu, Koren & Volinsky (2008), via the `implicit` package.
Scikit-learn has no ALS; `TruncatedSVD` is the nearest thing and it optimises
the wrong objective here, treating an unobserved pair as a zero rather than as an absence of evidence.
ALS instead treats purchase counts as **confidence** — nobody scored a Red Bull
four out of five, they bought it nine times — and `alpha` controls how fast
that confidence grows.

The item key is `gtin@daypart`. ALS takes no features, so the only way to give
it a clock is to put one in the key, and the clock matters: same-daypart
purchase pairs repeat the same SKU 18.0% of the time against 10.0% across
dayparts.

## The scored table

`score_table.py` writes `artifacts/customer_scores.parquet` — one row per
customer-moment in the held-out period (583,583 rows).

| Column | What it is |
|---|---|
| `shopper_id`, `store_id`, `ts`, `daypart`, `fuel_grade` | The moment |
| `prior_attach` | Their own attach rate before this fill; `-1` if new |
| `prob_entering` | `1 − Π(1 − pᵢ)` over the six categories. Derived, not modelled |
| `offer_category` | Stage one's pick |
| `personalised` | Whether ALS has a factor vector for them, or this is the popularity path |
| `<category>_propensity` × 6 | Category Relevance output |
| `top_skus` | `list[struct]`, ten rows, all from `offer_category` |
| `second_category` | The pair category — the complement, not the anchor |
| `second_category_skus` | `list[struct]`, all from `second_category`. **Empty** when nothing clears the complement gate |

The `top_skus` struct carries probabilities and nothing else:

| Field | Meaning |
|---|---|
| `affinity` | Raw ALS dot product — **or a unit count** on the fallback path. Read it with `personalised`; it is not one scale |
| `p_sku` | `P(this SKU \| this category)`, normalised within the category |
| `p_category` | `P(buys this category on this fill)` |
| `p_buys` | `p_category × p_sku` — `P(buys this exact SKU on this fill)` |

**No `unit_margin`, `funded_share` or `expected_value`.** The catalog's
margins are average price times an assumed 40%, because the PDI extract has
no cost column, so every dollar figure downstream of them is an assumption
wearing the costume of a measurement. They live in `negotiate.py`, which says
so out loud. The ranking is `p_buys`, which means the order is explainable
from the columns beside it.

### The second-item columns

`top_skus` is what they buy anyway, so it is the worst thing to discount.
`second_category_skus` is the other question: the complement a discount
might actually add to the basket. It is populated on **560,690 of 583,583
rows**; the rest have no pair that clears the gate, which includes every
alcohol-anchored row by rule.

| Field | Meaning |
|---|---|
| `affinity`, `p_sku` | As above, but ranked inside the pair category |
| `cohort_index` | The pair's rate among anchor regulars over the rate among equally frequent others. The **gate** |
| `excess` | The same comparison as a difference, in attach points. The **rank** |
| `headroom` | `1 − ` this shopper's own prior rate of that category |

**No `p_category` and no `p_buys` here.** The pair category usually sits
outside the six, so it has no Category Relevance model and there is no
`P(category)` for it. A product built from a placeholder would read as a
measurement. The two paths that fill this column — `BatchPairer.choose` and
the live `PairEngine` — are checked against each other in `validate.py`.

## The pairing question

The anchor pipeline predicts what a customer buys **anyway**, which makes it
exactly the wrong thing to discount. A promotion on the item they had already
decided to buy gives away margin and changes nothing. So the useful question
is the second one: given the drink they are going to buy, what *else* could
join the basket?

`pairing.py` answers it from the joint baskets. **40% of purchase baskets
already hold two or more categories**, and 46.5% of drink baskets are
drink-only — that second group is the addressable population.

Three measured quantities, and all three are needed:

| | What it is | What it is good for |
|---|---|---|
| `basket_lift` | `P(B \| A in basket) / P(B)` | Basket-level co-occurrence. Symmetric |
| `cohort_index` | B's attach rate among A-regulars ÷ the rate among **equally frequent** other shoppers | Gating: is this a real complement? |
| `excess` | the same comparison as a difference, in attach percentage points | **Ranking** |
| `headroom` | `1 −` this shopper's own observed rate of B | Don't discount what they already buy |

### Controlling exposure is the whole trick

Every cohort here is defined as "at least five purchase baskets", so every
cohort is a cohort of *heavy shoppers* and indexes high against the general
population on nearly everything. The first version of this analysis produced
nonsense for that reason — drink regulars appeared to over-index on frozen
food at 6.1×. Holding visit count constant on both sides divides that out:

```
drink regulars vs equally frequent others, per-basket attach rate
  Alternative Snacks      0.0602 vs 0.0346    index 1.74    excess 2.6pp
  Commissary              0.0316 vs 0.0193    index 1.64    excess 1.2pp
  Packaged Sweet Snacks   0.0669 vs 0.0474    index 1.41    excess 2.0pp
  Salty Snacks            0.1218 vs 0.0938    index 1.30    excess 2.8pp
  Candy                   0.1217 vs 0.1142    index 1.07    excess 0.8pp
  ...
  Cigarettes              0.0995 vs 0.2258    index 0.44
  Beer                    0.0373 vs 0.1893    index 0.20
```

### Gate on the index, rank on the excess

Neither works alone, and the failure modes are opposite:

- **Volume alone picks the popular item.** Candy is 18.2% of every second
  item in a drink basket and looks like the obvious pair. Its index is 1.07 —
  drink regulars buy it barely more than anyone else does. It is not a
  complement, it is just everywhere.
- **The index alone picks the tiny item.** Other Dairy indexes 1.32 on a
  0.55% attach rate. Flattering ratio, no volume.
- **The excess is in attach percentage points**, so it ranks by how much
  extra business the pairing is associated with, and it puts candy fifth.

### What the data actually says: there are two missions

| Anchor | Best complement | Verdict |
|---|---|---|
| Packaged Beverages | Alternative Snacks 1.74, Salty Snacks 1.30 | Pairs well |
| Salty Snacks | Alternative Snacks 1.97, Sweet Snacks 1.63, Candy 1.56 | Pairs well |
| **Beer** | Liquor 1.08 — and *nothing* else above 0.63 | **Never paired** (by rule) |
| **Cigarettes** | General Merchandise 2.46 — and nothing else above 0.65 | **Do not pair** |

The snack-and-drink categories are mutually complementary. The
beer/tobacco/lottery categories are mutually complementary *with each other*
and strongly **negative** against everything in the first group. A beer run
is not a snack run, and a customer on one mission is not a candidate for a
second item from the other. For a beer anchor the honest output is no pair
offer at all.

### Two caveats that matter

**None of this is causal.** Co-occurrence says these things appear together;
it does not say a discount on B *causes* a B purchase. That needs the same
holdout the rest of the pipeline is missing.

**The catalog now covers the complements.** `TARGET_CATEGORIES` was chosen
for the anchor problem and was the wrong list for this one: the three
strongest complements to a drink were not in `offer_catalog.csv` at all, so
the engine could name a category and then shrug. `SKU_MODEL_CATEGORIES` is
now `TARGET_CATEGORIES + PAIR_CATEGORIES`, adding Alternative Snacks (468
SKUs), Packaged Sweet Snacks (251) and Commissary (135).

This is safe for the anchor path by construction: stage one only considers
categories that have a Category Relevance model, and `train_models.py` fits
those over `TARGET_CATEGORIES` alone. The three new categories reach the
catalog and the ALS factors and nothing else — they can be the second item,
never the first.

**Alcohol is excluded from pairing on both sides** (`config.NOT_PAIRABLE`).
A beer anchor returns no pair offer and alcohol is never offered as a second
item. The measurement agrees — beer's only complement above 1.0 is Liquor at
1.08 — but the rule does not depend on that holding, and a pump screen that
answers "beer" with "and a bottle of liquor" is not a merchandising idea.
Note this is narrower than `NOT_OFFERABLE`: beer is still a legitimate
*anchor* offer, it just neither takes nor becomes a second item.


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
downstream. Probabilities are floored at `scoring.P_FLOOR` (1e-5) and
capped at 1 - that, in the one place both paths read it from.
