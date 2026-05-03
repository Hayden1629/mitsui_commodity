## Mitsui Pickup Notes

### Where you left off

Score progression: LSTM baseline → vanilla transformer hit **+0.0599 Sharpe** (after early stopping, smaller model, AdamW + warmup/cosine). Top private LB is 0.6 from a former commodities trader. Realistic target with this approach: 0.15-0.20.

### What the score actually is

Sharpe = mean(daily_rank_correlations) / std(daily_rank_correlations). Per day: rank predictions and targets, take Pearson of the ranks (= Spearman). Across days: mean/std of those daily numbers. Always log all three (mean, std, sharpe) when iterating so you can tell whether you're improving prediction quality or just smoothing variance.

### Dataset structure (from target_pairs.csv)

* 424 targets = 106 unique instrument pairs × 4 lags (1-4)
* 420 of 424 targets are spreads; only 4 are single-instrument
* Exchange grouping breakdown:
  * LME_US_Stock: 181
  * JPX_US_Stock: 87
  * FX_LME: 86
  * FX_JPX: 46
  * JPX_LME: 12
  * LME (pure): 8
  * FX (pure): 2
  * US_Stock (pure): 2
* LME metals (AH, ZS, CA, PB, plus NI/SN) appear in ~70-80 targets each
* JPX side dominated by Gold and Platinum futures
* FX is mostly a modifier across spreads, not a target itself

### Plan for tomorrow

#### Step 1: Feature engineering on raw input columns

Build these before any model training, on the full `train` dataframe:

1. **Lagged returns** for every price-like column at lags {1, 2, 5, 10, 20}: `(x_t - x_{t-k}) / x_{t-k}`
2. **Rolling volatility** (std of daily returns) at windows {5, 10, 20}
3. **Rolling z-scores** : `(x_t - rolling_mean) / rolling_std`
4. **Spread features for each spread target** : if target_N is `A - B`, add current `A - B` and its lags as features. Code stub already written:

python

```python
defadd_target_pair_features(df, target_info, feature_cols):
    new_features ={}
for _, row in target_info.iterrows():
ifnot row['is_spread']orlen(row['tickers'])!=2:
continue
ifnotall(t in feature_cols for t in row['tickers']):
continue
        col_name =f"spread_{row['target']}"
        new_features[col_name]= df[row['tickers'][0]]- df[row['tickers'][1]]
return pd.DataFrame(new_features, index=df.index)
```

Verify with `target_info['n_legs'].value_counts()` that all pairs are 1 or 2 legs. If any 3+, use the signed `(sign, ticker)` tuples from `parse_pair`.

#### Step 2: Regime features (shared across all models)

These get appended to every model's feature set:

* **Gold/silver ratio** (find the relevant JPX or LME columns)
* **Copper/gold ratio** (Dr. Copper signal, recession proxy)
* **Crude/Brent spread** if both exist in the data
* **USD index proxy** : average of major USD FX rates
* **Cross-sectional rolling vol** : mean of rolling vols across major equity tickers as a VIX proxy
* **Day-of-week, day-of-month dummies** (cheap)

Caveat from earlier: I oversold specific lead-lag claims (crude→airlines, copper→AUD). Directional relationships are real, exact lags are not something to hardcode. The ratios and spreads above are well-documented heuristics.

#### Step 3: Per-exchange-group models

Group targets by `exchange_key` → 8 groups. Train one **LightGBM multi-output** per group. Each gets:

* Features from the exchanges named in that group (e.g., LME_US_Stock model gets all LME features + all US_Stock features)
* Plus the engineered features (lagged returns, rolling vols, z-scores) on those columns
* Plus the spread features for targets in that group
* Plus the shared regime features

Each model predicts all targets in its group jointly across all 4 lags. Print per-group Sharpe on holdout, expect uneven performance (LME pure should be cleanest, FX-involved noisier).

#### Step 4: Ensemble and ship

* Average 3-5 random seeds per group model
* Optionally blend with the existing transformer (decorrelated errors)
* Keep early stopping logic from the transformer notebook

### Files/code already built (in old notebook)

* `parse_pair`, `get_exchange`, `get_instrument`, `target_info` builder
* `features_for_target(target, scope='direct'|'family'|'all')` lookup
* `TransformerPredictor` with early stopping, AdamW, warmup/cosine (keep for ensembling later)
* `mitsui_metric` matches the official scorer

### Reminders

* Holdout split: `date_id < 1827` train, rest holdout
* Use `ffill().fillna(0)` on features, never bfill (leaks future)
* For training targets, fillna(0); for scoring, use raw labels with NaN mask
* Watch val Sharpe in addition to val loss; loss going more negative is correlation improving on train
