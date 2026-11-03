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

# Model C (SKU choice) only earns its place where the tail is long. Top-100
# SKUs cover 45% of beverages but 86% of cigarettes -- there is nothing to
# personalise in the second case.
SKU_MODEL_CATEGORIES = ["Packaged Beverages (Non-Alcoholic)"]

# --- training ------------------------------------------------------------
# Train on everything before the cut, evaluate after it. Never split at
# random: a random split lets the model learn from next month to predict
# last month.
TRAIN_CUTOFF = "2026-04-01"
CALIB_FRACTION = 0.15      # of the training rows, held out to calibrate

# 36.1M visits does not fit comfortably in sklearn. Sample for Model A;
# Model B trains on converters only and is small enough to use whole.
SAMPLE_VISITS = 1_500_000
RANDOM_SEED = 20261103

# --- Model C: alternating least squares on implicit feedback -------------
# Hu, Koren & Volinsky (2008). Purchase counts are a CONFIDENCE signal, not
# a rating: nobody scored a Red Bull 4 out of 5, they just bought it nine
# times. alpha scales how much each observation raises that confidence.
ALS_FACTORS = 32
ALS_REGULARIZATION = 0.05
ALS_ALPHA = 20.0
ALS_ITERATIONS = 15
TOP_N_SKUS = 20            # candidates Model C hands to the re-ranker

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
