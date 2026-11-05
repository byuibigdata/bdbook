"""Shared configuration for the pump-screen pipeline.

Every constant that a reviewer might argue with lives here, so the argument
happens in one place instead of scattered through five scripts.
"""

from pathlib import Path

# --- paths ---------------------------------------------------------------
DATA = Path(__file__).resolve().parent.parent          # .../bdbook/data
ART = Path(__file__).resolve().parent / "artifacts"    # built tables + models
ART.mkdir(parents=True, exist_ok=True)

SRC = {
    "items": DATA / "transaction-items-all" / "*.parquet",
    "gtin": DATA / "master-gtin" / "*.parquet",
    "shopper": DATA / "shopper-id" / "*.parquet",
    "stores": DATA / "stores-information" / "*.parquet",
}

# --- the unit of analysis ------------------------------------------------
# A fuel visit is every purchase by the same shopper from BEFORE minutes
# before the fuel transaction to AFTER minutes after. Asymmetric because
# most people fuel first; 77% of in-store sales land within 5 minutes.
WINDOW_BEFORE = 15
WINDOW_AFTER = 45

# SERVICES is 99.5% card-processing fees charged on the fuel sale itself.
# Counting it as an in-store purchase leaks the fuel transaction into its
# own label and overstates the attach rate by about seven points.
NOT_A_PURCHASE = ["FUEL", "SERVICES"]

# --- dayparts ------------------------------------------------------------
# Three, not four: attach jumps from 9.8% to 13.2% at 05:00 and falls back
# after 23:00, and that overnight tail is only 3.2% of visits, so it rides
# with evening rather than earning a bucket of its own.
DAYPARTS = {"morning": (5, 9), "midday": (10, 14)}      # else -> "evening"


def daypart_of(hour: int) -> str:
    for name, (lo, hi) in DAYPARTS.items():
        if lo <= hour <= hi:
            return name
    return "evening"


# --- the offerable categories -------------------------------------------
# Six categories cover ~94% of converting visits. Modelling all 35 would
# spend effort on classes that never justify a slot on the screen.
TARGET_CATEGORIES = [
    "Packaged Beverages (Non-Alcoholic)",
    "Cigarettes",
    "Candy",
    "Salty Snacks",
    "Other Tobacco/Nicotine Products",
    "Beer",
]

# SKU Affinity (ALS) earns its place where the tail is long. Measured on the
# held-out period by validate_sku.py -- which is the only place these numbers
# should come from -- hit@10 is 34.3% against the best seller's 16.9% on the
# fills where ALS actually ran, a 2.0x lift, rising to 3.1x on beer. The
# personalised path covers 31.6% of scored fills.
#
# An earlier comment here claimed 16.2% vs 5.3% from a run that no longer
# exists and could not be reproduced. Do not quote a number for this model
# that validate_sku.py does not print.
#
# Note it is SKU and not brand: at brand level ALS scores 0.88x, i.e. WORSE
# than simply showing the best seller. With only a handful of brands that
# matter there is nothing personal left to learn; the long tail of SKUs is
# what gives the model something to do.
# The pairing question needs SKUs the anchor question never did. The three
# strongest complements to a drink -- Alternative Snacks 1.74, Commissary
# 1.64, Packaged Sweet Snacks 1.41 -- are not in TARGET_CATEGORIES, because
# that list was chosen for "what will they buy anyway" and these are
# categories almost nobody buys anyway. Without SKUs in the catalog the
# pairing engine can name the category and then shrug.
#
# Adding them here is safe for the anchor path: stage one only considers
# categories that have a Category Relevance model, and these do not. They
# reach the catalog and the ALS factors, and nothing else.
PAIR_CATEGORIES = [
    "Alternative Snacks",
    "Packaged Sweet Snacks",
    "Commissary & Other Packaged Products",
]

SKU_MODEL_CATEGORIES = TARGET_CATEGORIES + PAIR_CATEGORIES

# Categories we will not put on a pump screen whatever the arithmetic says.
# Tobacco carries the highest margin and the shortest SKU tail, so expected
# value alone pushes it at everyone -- including customers who have never
# bought it. Targeted tobacco promotion is restricted in most jurisdictions
# and is a reputational problem everywhere, so it is excluded as a business
# rule rather than left to the model.
NOT_OFFERABLE = ["Cigarettes", "Other Tobacco/Nicotine Products"]

# --- the pilot store set -------------------------------------------------
# Every model figure in the write-ups comes from these stores, not from the
# full estate. They were originally chosen by `head(25)` over whatever order
# the parquet scan happened to return, which is not reproducible: a rebuild
# silently drew a different 25 and moved every published number. The list is
# pinned here instead, recovered from the store_id encoding of the models
# that produced the quoted figures.
#
# Two reasons the models are not fitted on all 708 stores:
#   - HistGradientBoosting caps a categorical feature at 255 levels, and
#     `store_id` is one of ours. The full estate raises a ValueError.
#   - The exploration numbers (36.1M visits, 12.8% attach) ARE full-estate.
#     Only the fitted models are the pilot subset. Do not mix the two.
PILOT_STORES = [
    "1696", "1878", "1881", "1882", "8219", "8228", "8230", "11794",
    "11934", "12272", "12880", "13065", "13413", "13525", "14292",
    "15279", "33642", "36216", "42426", "42427", "42475", "42748",
    "42937", "45996",
]

# --- training ------------------------------------------------------------
# Train on everything before the cut, evaluate after it. Never split at
# random: a random split lets the model learn from next month to predict
# last month.
TRAIN_CUTOFF = "2026-04-01"
CALIB_FRACTION = 0.15      # of the training rows, held out to calibrate

# 36.1M visits does not fit comfortably in sklearn, so the Category
# Relevance models train on a sample. They now fit on EVERY visit rather
# than on converters only, since the separate attach model was dropped.
SAMPLE_VISITS = 1_500_000
RANDOM_SEED = 20261103

# --- SKU Affinity: alternating least squares on implicit feedback --------
# Hu, Koren & Volinsky (2008). Purchase counts are a CONFIDENCE signal, not
# a rating: nobody scored a Red Bull 4 out of 5, they just bought it nine
# times. alpha scales how much each observation raises that confidence.
ALS_FACTORS = 32
ALS_REGULARIZATION = 0.05
ALS_ALPHA = 20.0
ALS_ITERATIONS = 15
TOP_N_SKUS = 20            # candidates SKU Affinity hands to the re-rank

# --- economics -----------------------------------------------------------
# The PDI data has no cost column, so these are ASSUMPTIONS and every
# dollar figure downstream inherits them. See offer_catalog.csv.
ASSUMED_MARGIN_RATE = 0.40

# Funding is a SHARE of the discount, not a yes/no. A brand often pays all
# of it, and sometimes the retailer and the brand split the cost:
#
#     our_cost        = discount x (1 - funded_share)
#     effective margin = unit_margin - our_cost
#
# At funded_share = 1 the margin per unit is untouched, so the only question
# left is whether the offer sells one more unit. At 0 the same $0.90 discount
# against a ~$1.30 margin needs the purchase rate to more than triple.

# Screen space costs nothing at the margin, so the honest threshold is zero:
# show an offer whenever it is expected to make money, and nothing when every
# candidate loses. In practice that means vendor-funded offers almost always
# clear the bar (our cost is 0) and self-funded ones at a $0.90 discount on a
# ~$1.30 margin almost never do.
MIN_EXPECTED_VALUE = 0.0


# ---------------------------------------------------------------- pairing
# The pairing question is a different question: not "what will they buy",
# but "what SECOND thing could a discount add to the basket". The item they
# came for needs no discount -- that is the one promotion that is certain to
# cost margin and change nothing.
#
# A cohort is the shoppers for whom one category dominates. Comparing them
# against EQUALLY FREQUENT other shoppers is the whole trick: a cohort
# defined by "at least five baskets" indexes high on everything, because it
# is a cohort of heavy shoppers. Divide exposure out and the mission
# structure appears.
PAIR_MIN_VISITS = 5          # purchase baskets needed to join a cohort
PAIR_DOMINANT = 0.60         # share of their baskets holding the anchor
PAIR_MIN_SUPPORT = 200       # cohort baskets containing the pair category
# Below this the pair is not a complement, it is just a popular item. Candy
# is the cautionary case: 17.9% of all second items in a drink basket, and
# an index of 1.07.
PAIR_MIN_INDEX = 1.05

# Alcohol is out of the pairing question on both sides. Beer's only measured
# complement is Liquor at 1.08, and a pump screen that answers "beer" with
# "and a bottle of liquor" is not a merchandising idea, it is a liability.
# Beer regulars buy drinks at 0.20x and cigarettes at 0.53x -- a beer run is
# its own mission and pairs with nothing we would want to discount, so the
# honest output for a beer anchor is no pair offer at all.
#
# Note this is narrower than NOT_OFFERABLE: beer remains a legitimate ANCHOR
# offer, it just does not take or become a second item.
NOT_PAIRABLE = ["Beer", "Wine", "Liquor"]
