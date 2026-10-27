# Mid-semester plan

Proposed updates to the slides, drawn from the two whiteboard photos in this folder
(`IMG_1877.jpeg`, the schedule; `IMG_1876.jpeg`, the training model). Written 10/23.

## 1. What the photos say

### The schedule (IMG_1877)

| Date | On the board | Weekday |
|---|---|---|
| 10/27 | Written challenge / review coding challenge / group time | Tue |
| 10/29 | Present on simple features / basket of goods practices | Thu |
| 11/3 | Basket of goods finish / complex features | Tue |
| 11/5 | Data types written challenge | Thu |
| 11/10 | Present complex features | Tue |
| 11/12 | 2nd take-home | Thu |
| 11/17 | **Streamlit and apps, taught by Mighty Oaks** · written challenge PySpark 2 | Tue |
| 11/19 | **Docker, taught by Fabrice and Friends** · ML take-home | Thu |
| 11/24 | Team day | Tue |
| 11/26 | Thanksgiving — no class | Thu |
| 12/1 | ML challenge (3rd PySpark verbal) | Tue |
| 12/3 | Google Cloud | Thu |
| 12/8 | **Present model results** | Tue |
| 12/10 | **Open** — held as buffer | Thu |
| 12/15 | Docker / Google Cloud / app challenge | Tue |

This confirms the Tuesday/Thursday pattern the day decks have been assuming.

### The training model (IMG_1876)

How a short training should be built:

```
Frame the need  →  Navigate to details  →  Give examples  →  Let them practice  →  Review / reflect
└────────── slides ──────────┘            └────────── code ──────────┘
└──────────────────────── industry ────────────────────────┘
```

Plus two reference points for what a good module looks like: **DataCamp** (modules of
short videos paired with coding) and **AI study guides**.

## 2. Where the decks stand

- **Day decks:** `day01` through `day13`, covering 9/15 through 10/22.
- **Topic decks:** `env_variables`, `association_rules` (Baskets of Goods), `basket_models`,
  `customer_segmentation`.
- **Planning notes:** `_old_agendas/README.md` still holds my *reconstructed* schedule for the
  back half of the semester. The board replaces it.

## 3. Proposed day decks

| Date | Deck | What it carries |
|---|---|---|
| 10/27 | `day14_written_review.qmd` ✅ | Written challenge, review of the coding challenge, group time |
| 10/29 | `day15_simple_features.qmd` ✅ | Team presentations on simple features, then **Baskets of Goods** |
| 11/3 | `day16_complex_features.qmd` | Finish basket of goods, introduce complex features |
| 11/5 | `day17_data_types.qmd` | Data types written challenge |
| 11/10 | `day18_complex_presentations.qmd` | Teams present complex features |
| 11/12 | `day19_takehome2.qmd` | Second take-home challenge |
| 11/17 | `day20_apps.qmd` | **Mighty Oaks teach Streamlit and apps** · written challenge PySpark 2 |
| 11/19 | `day21_docker.qmd` | **Fabrice and Friends teach Docker** · ML take-home |
| 11/24 | `day22_team_day.qmd` | Short — team day |
| 12/1 | `day23_ml_challenge.qmd` | ML challenge, third PySpark verbal |
| 12/3 | `day24_google_cloud.qmd` | Google Cloud |
| 12/8 | `day25_present_results.qmd` | **Present model results** |
| 12/10 | `day26_open.qmd` | **Open** — buffer, or a second round of team trainings |
| 12/15 | `day27_app_challenge.qmd` | Docker / Google Cloud / app challenge |

**Note:** 11/26 is Thanksgiving, so no deck. Whether 11/26 needs a "no class" placeholder like
`day08_no_class.qmd` is your call.

## 4. Proposed topic decks

### New, in the order the schedule needs them

| Deck | Needed by | Covers |
|---|---|---|
| `topic_data_types.qmd` | 11/5 | Spark types, casts that silently fail, nulls, schema drift — the written-challenge material |
| `topic_complex_features.qmd` | 11/3 | Window functions, lags, rolling windows, leakage. Pairs with the book's temporal feature chapter |
| `topic_apps.qmd` | 11/17 | Streamlit / Marimo, what makes an app usable by a non-technical person |
| `topic_docker.qmd` | 11/19 | Images, containers, volumes, the compose file from the first challenge |
| `topic_google_cloud.qmd` | 12/3 | Build, push, run, cost, and keeping keys out of the image |
| `topic_verbal_prep.qmd` | before 12/1 | What a PySpark verbal asks and how to prepare for it |

### Already built, and where they land

- **`association_rules`** (Baskets of Goods) → 10/29 and 11/3
- **`basket_models`** → the ML challenge, 12/1
- **`customer_segmentation`** → the ML challenge, 12/1, and the analysis half of the project
- **`env_variables`** → already used on 11/17 and 12/3 for app and cloud keys

## 5. The training model deserves its own deck

The second photo is a teaching method, and every team owes a training. Proposal:

**`topics/build_a_training.qmd`** ✅ **built** — covering:

1. **Frame the need.** Why the class should care, in one slide. Industry framing, not academic.
2. **Navigate to details.** Enough background to follow along, no more.
3. **Give examples.** Working code the class can read.
4. **Let them practice.** A hands-on activity — the part teams skip and shouldn't.
5. **Review / reflect.** What they should take away, and where to look next.

With two framing points from the board:

- **Slides carry steps 1–2; code carries steps 3–4.** Teams usually over-invest in slides.
- **DataCamp is the model:** short video, immediate coding, repeat. **AI study guides** are a
  reasonable aid for building the material, not a substitute for the practice step.

Alongside it, a **starter file** teams copy (`topics/_training_template.qmd`) ✅ **built**, pre-structured with those five sections, so a team's deck arrives in the course brand with no theme setup.

## 6. Housekeeping

- **Replace** the proposed schedule in `_old_agendas/README.md` with the board's dates, and mark
  the earlier version as superseded.
- **Update the class repo** (`FA26_hathaway`): its challenge table still has empty due dates for
  everything after the easy feature challenge, and `agenda.md` still holds last semester's
  January–April schedule.
- **Fill in dates** on the slides as they firm up: the 2nd take-home, written challenge PySpark 2,
  the ML challenge and the app challenge all appear on the board without due dates.
- **Name the teaching slots.** The board has *(Mighty Oaks)* on 11/17 and *(Fabric)* on 11/19,
  which look like the first two team trainings. The remaining teams need dates.

## 7. Decisions (was: open questions)

| Question | Decided |
|---|---|
| What happens 12/10? | **Open.** Held as buffer. |
| What is the ✱ on 12/8? | **Present model results.** |
| Who teaches 11/17? | **Mighty Oaks** — Streamlit and apps. |
| Who teaches 11/19? | **Fabrice and Friends** — Docker. |

Still open:

- **Verbals 1 and 2 need dates.** Only the third is on the board, on 12/1.
- **The remaining teams need teaching slots.** Two of them are set; 12/10 is the obvious home for
  a second round if more teams want a date.
- **Due dates** for the 2nd take-home, written challenge PySpark 2, the ML challenge and the app
  challenge are not on the board yet.

## 8. Housekeeping status

- ✅ `_old_agendas/README.md` now carries the board's real dates; the earlier reconstruction is
  marked superseded, and every built deck is listed.
- ⬜ **The class repo still needs updating** (`FA26_hathaway`): empty due dates after the easy
  feature challenge, and `agenda.md` still holds last semester's January–April schedule. That is a
  separate repo, so it needs a push from you, or a go-ahead from you for me to open a PR.
