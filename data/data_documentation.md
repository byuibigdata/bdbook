Overview
PDI Technologies (formerly Skupos) provides SKU-level point-of-sale (POS) data from tens of thousands of independent convenience stores across the United States, covering roughly 80% of the independent convenience-store market. Through its acquisition of Skupos, PDI maintains the largest panel of POS data from independent convenience stores in the United States.&#x20;

The PDI Convenience Store Transaction Dataset is composed of nine relational data files that share unique identifiers across stores, transactions, payments, items, discounts, and shoppers, enabling researchers to assemble a full picture of in-store consumer purchases at SKU granularity.

Standard join keys include: STORE_ID, TRANSACTION_SET_ID, TRANSACTION_ITEM_ID, PAYMENT_ID, and GTIN.

Data Description
The dataset consists of nine relational tables forming a star-with-relations schema:

Stores Information
One row per convenience store in the PDI panel with descriptive metadata: store name, chain affiliation, address (street, city, state, ZIP), latitude/longitude, chain size, and the start date for the store joining PDI's panel. Also tracks the start week beyond which the store's transaction data is continuous (no delivery gaps). This is the canonical "stores" dimension table that every transaction-level table joins to via STORE_ID.&#x20;

Observation Level: One row per store in the PDI panel, keyed by STORE_ID.

Stores Status
Stores dimension extended with operational status flags. Adds STORE_FLAG and ACTIVE_STATUS columns indicating whether a store is currently active, on hold, or otherwise excluded from delivery. Otherwise mirrors the Stores Information schema. Use this table to filter out inactive stores when computing time-series aggregates.&#x20;

Observation Level: One row per store, keyed by STORE_ID. Same grain as Stores Information.&#x20;

Master GTIN
The canonical SKU dimension. One row per 14-digit GTIN with PDI-curated product attributes: category, subcategory, manufacturer parent, manufacturer, brand, product type, sub-product type, flavor, unit size, pack size, package, and a descriptive product name. The reference table for joining transaction-level item lines back to product-level metadata.&#x20;

Observation Level: One row per unique GTIN (14-digit Global Trade Item Number).&#x20;

Transaction Sets
Transaction header / basket-level table. One row per transaction at a store, capturing transaction timestamp, POS vendor type (Verifone, Gilbarco, Clover, NCR), payment type, subtotal, taxable amount, tax amount, tax rate, grand total, tender received, and tender given (change). Parent table for line-item, discount, and payment children which join via TRANSACTION_SET_ID.&#x20;

Observation Level: One row per completed transaction (basket), keyed by TRANSACTION_SET_ID.&#x20;

Transaction Items
Line-item-level transaction table. One row per item line within a transaction with GTIN, POS-assigned description, unit price, unit quantity, discount amount applied at the line, taxable amount, tax rate, grand total, and scan vs. non-scan capture method. Non-scan items carry NACS category / subcategory / detail labels.Joins to Master GTIN via GTIN for product metadata; joins to Transaction Sets via TRANSACTION_SET_ID for basket context; child of the transaction set.&#x20;

Observation Level: One row per line item within a transaction, keyed by TRANSACTION_ITEM_ID.&#x20;

Transaction Items Daily Aggregation
Pre-aggregated daily summary of transaction items. One row per (store, GTIN or non-scan category, calendar date) with quantity sold, transaction count, total revenue amount, quantity sold under a discount, and transaction count with a discount. Includes date dimensions (day of week, week, calendar month, calendar year) for easier time-series analysis without joining against the line-item table. Designed for analyses that don't need individual line-item resolution.&#x20;

Observation Level: One row per unique combination of STORE_ID + GTIN (or non-scan category) + DATE, keyed by ID.&#x20;

Payments
Payment-method records associated with transactions. One row per payment instrument applied to a transaction, capturing payment type (cash, debit, credit, etc.), payment entry mode (swipe, etc.), card type, merchant ID, terminal ID, payment locale, currency, tender amount, and acquirer reference / card auth codes when applicable. A single transaction may have multiple payment rows (split tender). Joins to Transaction Sets via TRANSACTION_SET_ID.&#x20;

Observation Level: One row per payment instrument used on a transaction, keyed by PAYMENT_ID.&#x20;

Discounts
Discount-line records applied to transaction items. One row per discount applied. Includes discount type code, discount name (parsed from t-log), discount number, quantity, adjustment amount, discount amount, and grand total. Discounts have a 1-to-many (1:N) relationship with transaction items — multiple discounts can be applied to the same line item. Joining Discounts to Transaction Items or Transaction Sets can fan out rows; aggregate discounts to the item-line grain first if total-recoverable-discount per line is the goal.&#x20;

Observation Level: One row per discount applied to a transaction line item, keyed by DISCOUNT_ID.&#x20;

Shopper ID
A privacy-preserving identifier table that attempts to link transactions to unique shoppers. One row per payment containing a SHOPPER_ID string assigned by PDI's identity-resolution methodology, plus the parent PAYMENT_ID and TRANSACTION_SET_ID. Enables longitudinal shopper-level analysis — repeat-visit rates, basket-size distributions per shopper, brand loyalty — without exposing personally identifiable information.

Observation Level: One row per payment with a resolved shopper, keyed by PAYMENT_ID. Not all payments have a resolved shopper; coverage depends on the underlying identity signal (e.g. card-on-file).&#x20;

Coverage
Geography: United States; tens of thousands of independent convenience stores covering roughly 80% of the independent convenience-store market.&#x20;

Time: Continuous SKU-level POS coverage from 2023 onward, with most-recent-load timestamps in the 2025-2026 window. New transactions ingested incrementally as POS t-logs are delivered.&#x20;

Methodology
All PDI datasets are sourced from PDI Technologies' integrated POS data feed across its panel of tens of thousands of independent U.S. convenience stores, parsed from in-store POS transaction log (t-log) files across four supported vendors (Verifone, Gilbarco, Clover, NCR). Dimension tables (Stores Information, Stores Status, Master GTIN) are curated by PDI through panel management and product-hierarchy sources with NACS category mappings; transaction-level tables (Transaction Sets, Transaction Items, Payments, Discounts, Shopper ID) capture every line, payment, and discount as recorded by the POS. Transaction Items Daily Aggregation id derived nightly from the line-item table for faster time0series analysis without scanning individual transactions.&#x20;

Additional Notes
Relational Structure: The nine tables form a star-with-relations schema. Standard join paths include:

Stores Information / Stores Status ← STORE_ID → Transaction Sets, Transaction Items, Payments, Discounts, Transaction Items Daily Aggregation
Transaction Sets ← TRANSACTION_SET_ID → Transaction Items, Payments, Discounts, Shopper ID
Transaction Items ← TRANSACTION_ITEM_ID → Discounts (1:N)
Payments ← PAYMENT_ID → Shopper ID (1:1)
Master GTIN ← GTIN → Transaction Items, Transaction Items Daily Aggregation
FAQs
Are large chains such as 7-Eleven, Buc’ees, Pilot etc. included in these data?
The data that PDI is providing is primarily from the category of independently owned convenience stores, which is the largest segment of the c-store market in the US. The data for some of the large chains mentioned, 7-Eleven, Buc’ees, Pilot etc., are not available due to data rights restrictions, so they are therefore not in the data.

How is ShopperID determined?
ShopperID: Created by hashing the concatenated values of first 6 digits of the card (BIN), the last 4 digits of the card, and the state abbreviation of the store where a purchase was made.
Cash and EBT payments are not included in the models.
Why are cash payments with associated ShopperIDs?
Possible that this was the result of split payment where multiple payment types were used.
How are lottery items handled?
PDI notes that these lottery items appear to be seasonal, and so without continuous GTIN mapping it is reasonable to see a decline in the number to GTINs in that library.
The last update to the library was the beginning of the year (2025), the decline is expected.
How can I filter lottery transactions that are mislabeled?
PDI confirms that filtering lottery transactions to the whole dollar amount can remove miscategorized GTINs.
PDI also suggests to filter those transactions associated with a discount to remove miscategorized GTINs.
What are the mappings for DISCOUNT_TYPE?
The mappings are as follows: 0 = Mix & Match, 1 = Combo, 2 = Manager, 3 = Loyalty, 4 = Cashier Discount, 5 = Cashier Price Override, 6 = Retailer Discount, 7 = Retailer Promo, 8 = Retailer Price Override, 9 = Missing Discount
---

# Column Details

Transcribed from PDI's published data dictionary pages; the source screenshots are in
`column_details_images/`. These are PDI's own column descriptions, reproduced as text so
they can be searched, diffed, and fed to tooling.

Note on scope: PDI documents nine tables. This extract contains seven of them — **Transaction
Sets** (the basket header) and **Transaction Items Daily Aggregation** were not delivered.
Their absence is why there is no transaction-grain table to join through.

## Transaction Items

Also published as *Historical Transaction Items*, with an identical column list.

| Column Name | Description |
|---|---|
| TRANSACTION_ITEM_ID | Primary key of the transaction items table. |
| TRANSACTION_SET_ID | Primary key of the transaction sets table. |
| STORE_ID | Primary key of the stores table. |
| DATE_TIME | Date time of transaction which is from t-log file. This is the wall clock time for the store. |
| GTIN | A 14-digit product identifier. |
| POS_DESCRIPTION | The POS-assigned item description. |
| UNIT_PRICE | Unit price. |
| UNIT_QUANTITY | The unit quantity. |
| DISCOUNT_AMOUNT | Discount amount applied to the items. |
| TAXABLE_AMOUNT | Taxable amount. 0 for gas. |
| TAX_RATE | Shows the tax rate but, if there are different items with different tax rates in the same transaction, it is showing only one of them. |
| GRAND_TOTAL_AMOUNT | Total amount paid including tax. Subtotal_amount + tax_amount. |
| SCAN_TYPE | Relates to how the product details were captured (Scanned, Manual entry, etc.). |
| NONSCAN_CATEGORY | The NACS category. |
| NONSCAN_SUBCATEGORY | The NACS subcategory. (Mapped attribute: POI Sub-Category) |
| NONSCAN_DETAIL | The NACS detail. |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

`CREATED_AT` and `UPDATED_AT` are documented but **absent from the delivered parquet files**.

## Payments

| Column Name | Description |
|---|---|
| PAYMENT_ID | Primary key of the payments table. |
| TRANSACTION_SET_ID | Primary key of the transaction sets table. |
| STORE_ID | Primary key of the stores table. |
| POS_TYPE_ID | Shows the type of Vendor that comes from pos_types table. 1 - Verifone, 2 - Gilbarco, 3 - Clover, 4 - NCR |
| DATE_TIME | Date time of transaction which is from t-log file. This is the wall clock time for the store. |
| PAYMENT_NUMBER | Corresponds for payment_number. Only for Clover. |
| PAYMENT_TYPE | Transaction payment type (debit, cash, etc.). |
| PAYMENT_ENTRY | Corresponds to payment entry type (Swipe, etc.). |
| CARD_TYPE | Corresponds to card type (Debit, Personal, etc.). |
| MERCHANT_ID | Merchant id parsed from t-log file. Only for Verifone. |
| TERMINAL_ID | Terminal id parsed from t-log file. Only for Verifone. |
| PAYMENT_LOCALE | Currency type of payment. |
| CURRENCY_LOCALE | Local currency value (USD, Dollars, etc.). |
| TENDER | The amount the customer gives as payment or gets back as change. |
| CHANGE | Parsed from t-log and value indicated whether a customer get back change. |
| ACQUIRER_REFERENCE_NUMBER | The acquirer reference number parsed from t-log. For Verifone, Gilbarco and NCR. |
| CARD_AUTH_CODE | Card authorization code. |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

`ACQUIRER_REFERENCE_NUMBER`, `CARD_AUTH_CODE`, `CREATED_AT` and `UPDATED_AT` are documented
but **absent from the delivered parquet files** — likely stripped as cardholder data.

## Discounts

| Column Name | Description |
|---|---|
| DISCOUNT_ID | Primary key of the discounts table. |
| TRANSACTION_ITEM_ID | Primary key of the transaction items table. |
| TRANSACTION_SET_ID | Primary key of the transaction sets table. |
| STORE_ID | Primary key of the stores table. |
| POS_TYPE_ID | Shows the type of Vendor that comes from pos_types table. 1 - Verifone, 2 - Gilbarco, 3 - Clover, 4 - NCR |
| DATE_TIME | Date time of transaction which is from t-log file. This is the wall clock time for the store. |
| DISCOUNT_TYPE | Correspondence to discount type is fetched from t-log file. |
| DISCOUNT_NAME | Name of the item that discount applied fetched from t-log file. |
| DISCOUNT_NUMBER | Discount number. |
| QUANTITY | The discount quantity. |
| ADJUSTMENT_AMOUNT | Discount adjustment amount. |
| DISCOUNT_AMOUNT | Discount amount applied to the items. |
| GRAND_TOTAL_AMOUNT | Total amount paid including tax. Subtotal_amount + tax_amount. |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

### DISCOUNT_TYPE mappings

| Code | Meaning |
|---|---|
| 0 | Mix & Match |
| 1 | Combo |
| 2 | Manager |
| 3 | Loyalty |
| 4 | Cashier Discount |
| 5 | Cashier Price Override |
| 6 | Retailer Discount |
| 7 | Retailer Promo |
| 8 | Retailer Price Override |
| 9 | Missing Discount |

## Master GTIN

| Column Name | Description |
|---|---|
| GTIN | A 14-digit product identifier. |
| CATEGORY | The PDI Technologies-defined item category. |
| SUBCATEGORY | The PDI Technologies-defined item subcategory. |
| MANUFACTURER_PARENT | The parent manufacturer. |
| MANUFACTURER | The manufacturer. |
| BRAND | The brand. |
| PRODUCT_TYPE | The PDI Technologies-defined product type. |
| SUB_PRODUCT_TYPE | The PDI Technologies-defined subproduct type. |
| FLAVOR | The flavor. |
| UNIT_SIZE | The unit size. |
| PACK_SIZE | The pack size. |
| PACKAGE | The item packaging. |
| PRODUCT_DESCRIPTION | The PDI Technologies-assigned item description. |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

Note the two different taxonomies: `master_gtin.CATEGORY` is **PDI-defined**, while
`transaction_items.NONSCAN_CATEGORY` is **NACS**. Coalescing them mixes two vocabularies.

## Shopper ID

| Column Name | Description |
|---|---|
| PAYMENT_ID | Primary key of the payments table. |
| TRANSACTION_SET_ID | Primary key of the transaction sets table. |
| SHOPPER_ID | A PDI Technologies-defined ID that attempts to identify unique shoppers. |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

`CREATED_AT` and `UPDATED_AT` are documented but **absent from the delivered parquet files**.

## Stores Information

| Column Name | Description |
|---|---|
| STORE_ID | Primary key of the stores table. |
| STORE_NAME | The store name. |
| STORE_CHAIN_ID | Unique store chain identifier. |
| STORE_CHAIN_NAME | The store chain name. |
| STREET_ADDRESS | The street address. |
| CITY | The city. |
| STATE | The state. |
| ZIP_CODE | The zip code. |
| LATITUDE | Latitude. |
| LONGITUDE | Longitude. |
| START_DATE | The start date for the store with PDI Technologies. |
| START_WEEK_CONTINUOUS_DATA | The start week for the store with PDI Technologies beyond which the data is continuous (i.e. no gaps). |
| CHAIN_SIZE | The number of stores in the chain. |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

## Stores Status

| Column Name | Description |
|---|---|
| STORE_ID | Primary key of the stores table. |
| STORE_NAME | The store name. |
| STORE_CHAIN_ID | Unique store chain identifier. |
| STORE_CHAIN_NAME | The store chain name. |
| STORE_FLAG | Store flag |
| ACTIVE_STATUS | Active Status |
| STREET_ADDRESS | The street address. |
| CITY | The city. |
| STATE | The state. |
| ZIP_CODE | The zip code. |
| LATITUDE | Latitude. |
| LONGITUDE | Longitude. |
| CHAIN_SIZE | The number of stores in the chain. |
| START_DATE | The start date for the store with PDI Technologies. |
| START_WEEK_CONTINUOUS_DATA | The start week for the store with PDI Technologies beyond which the data is continuous (i.e. no gaps). |
| CREATED_AT | Date time when this record was created. |
| UPDATED_AT | Date time when this record was last updated. |

PDI's narrative says these flags indicate "whether a store is currently active, on hold, or
otherwise excluded from delivery", but the two columns are not formally defined. Observed
combinations across all 36,912 stores: `A`/`TRUE` 13,747, `A`/`FALSE` 4,618, `I`/`FALSE`
18,547. No store is `I`/`TRUE`.
