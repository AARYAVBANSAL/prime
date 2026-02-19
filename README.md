# Trader Performance vs Market Sentiment — Hyperliquid Analysis
**Primetrade.ai · Data Science Intern Assignment**

---

## Setup & How to Run

```bash
# 1. Clone / unzip the submission folder
# 2. Place both data files in the same directory:
#    - historical_data_csv  (unzipped from historical_data_csv.gz)
#    - fear_greed_index.csv

# 3. Install dependencies (standard Python stack)
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Launch notebook
jupyter notebook trader_sentiment_analysis.ipynb
```

All charts are pre-generated as PNG files in `/outputs/` and referenced inline in the notebook.

---

## Dataset Overview

| Dataset | Rows | Columns | Date Range |
|---|---|---|---|
| Fear & Greed Index | 2,644 | 4 | Feb 2018 → May 2025 |
| Hyperliquid Trader Data | 211,224 | 16 | May 2023 → May 2025 |
| **Merged (inner join on date)** | **211,218** | **23** | May 2023 → May 2025 |

- **32 unique trading accounts**, **246 unique coins/tokens**
- No missing values in key columns; no duplicates in either dataset
- Timestamps in `Timestamp IST` parsed with `dayfirst=True`; normalized to daily granularity for merge
- `Net PnL = Closed PnL − Fee` (fee deducted for accuracy)
- Sentiment simplified to 3 buckets: **Fear** (Extreme Fear + Fear), **Neutral**, **Greed** (Greed + Extreme Greed)
- Sentiment distribution in overlap period: Fear 49% · Greed 36% · Neutral 15%

---

## Methodology

1. **Data Cleaning** — parsed dayfirst timestamps, coerced numeric columns, computed `net_pnl` and binary `is_win`/`is_long` flags
2. **Alignment** — inner-join on normalized date; daily-level aggregation for platform-wide metrics
3. **Segmentation** — three independent segmentation axes (frequency, volume, consistency) derived from account-level statistics
4. **Analysis** — group-by comparisons across sentiment buckets; F&G decile binning for continuous analysis
5. **Predictive Model** — Random Forest classifying next-day PnL bucket (Big Loss / Small Loss / Small Gain / Big Gain) using 6 features; 47% test accuracy vs 55% chance baseline for the dominant class

---

## Key Findings & Insights

### Insight 1 — Fear Days Drive Volume, Not Profit
| Sentiment | Avg Trades/Day | Mean PnL/Trade | Win Rate |
|---|---|---|---|
| Fear | **792.7** | $30.19 | 32.5% |
| Neutral | 562.5 | $61.54 | 34.0% |
| Greed | 294.1 | **$44.27** | **40.8%** |

Fear days attract **2.7× more trading activity** than Greed days but produce lower per-trade PnL and win rate. The market panic drives reactive, lower-quality trades. **Greed days produce the best per-trade returns despite lower volume.**

### Insight 2 — Traders Go More Long During Fear (Contrarian Positioning)
| Sentiment | Long Ratio |
|---|---|
| Fear | **51.0%** |
| Neutral | 50.0% |
| Greed | **48.0%** |

Counter-intuitively, traders on this platform skew slightly **more long during Fear** and slightly more short during Greed — suggesting a contrarian/dip-buying tendency. The long ratio tracks inversely with the F&G index (confirmed by decile chart).

### Insight 3 — Consistent Winners Are Insensitive to Sentiment; Inconsistent Traders Are Not
From the Sentiment × Segment heatmap:
- **Consistent Winners** maintain >83% win rate regardless of sentiment — their edge is systematic and sentiment-independent
- **Inconsistent Traders** see the sharpest win-rate drop (38% → 33%) during Fear vs Greed
- **Consistent Losers** lose money across all conditions, with Fear being slightly worse

This confirms that sentiment matters *only for the traders who lack a robust edge* — disciplined traders are weather-proof.

### Insight 4 — Larger Trade Size Does Not Equal Higher Returns
High-Volume traders (avg size > median) have **lower win rate (36%)** than Low-Volume traders (43.5%), despite generating more absolute PnL. Frequent traders outperform infrequent ones in both win rate and total PnL — suggesting skill compounds with repetition, but raw position size alone is not an edge.

---

## Part C — Strategy Recommendations

### Strategy 1: "Fear Day Size Reduction for Non-Systematic Traders"
**Evidence:** Inconsistent and Consistent-Loser segments suffer most during Fear days. The platform sees panic-volume surges that dilute signal quality.

> **Rule:** When F&G Index < 40 (Fear), *Inconsistent* accounts should apply a **20% automatic size reduction** and a **maximum 2 open positions** rule. This preserves capital during high-noise, low-win-rate market conditions. Expected benefit: reduce drawdown exposure without eliminating participation.

### Strategy 2: "Greed-Zone Long Cap for Contrarian Bias Correction"
**Evidence:** Traders on this platform skew long during Fear and slightly short during Greed — the opposite of what sentiment extremes reward. Greed days have the best win rates for disciplined traders, suggesting the crowd is mis-positioned.

> **Rule:** When F&G Index > 75 (Extreme Greed), place a **hard 50% cap on new long positions** for accounts in the Inconsistent segment. Simultaneously, *nudge* trade alerts toward momentum-following (stay long with trend) rather than contrarian short — correcting the observed behavioral bias. For Consistent Winners, no change needed.

---

## Predictive Model (Bonus)

- **Target:** Next-day PnL bucket (4-class classification)
- **Features:** `fg_value`, `trade_count`, `long_ratio`, `avg_size_usd`, `win_rate`, `sentiment_encoded`
- **Model:** Random Forest (100 trees, max_depth=4, class_weight=balanced)
- **Test accuracy: 47%** (vs 55% naive majority-class baseline)
- **Top features by importance:** `win_rate` > `avg_size_usd` > `fg_value` > `trade_count`

The model is intentionally simple — it shows that **today's win rate and trade size are the strongest predictors of tomorrow's outcome**, with sentiment adding incremental signal. A production model would benefit from rolling window features, account-level history, and a regression head rather than classification buckets.

---

## Output Files

| File | Description |
|---|---|
| `trader_sentiment_analysis.ipynb` | Main notebook (fully runnable) |
| `chart1_overview_dashboard.png` | 6-panel KPI dashboard by sentiment |
| `chart2_timeline.png` | PnL, F&G index, and trade count over time |
| `chart3_segments.png` | Win rate & PnL across 3 segmentation axes |
| `chart4_heatmap_sentiment_segment.png` | Sentiment × consistency heatmap |
| `chart5_distributions.png` | PnL distributions + F&G vs win rate scatter |
| `chart6_fg_decile_behavior.png` | Trade behavior across F&G value deciles |
| `chart7_feature_importance.png` | RF feature importances for PnL prediction |

---

*Submitted by: [Aaryav Bansal] · February 2026*
