# VN30 Stock Price Forecasting — PySpark Modeling Pipeline

## 1. Overview

This pipeline forecasts closing prices for all **30 VN30 stocks** simultaneously
using historical daily trading data from 2018 to 2026.

Five models are compared under a unified evaluation framework:

| Group | Model | Engine |
|---|---|---|
| Statistical | ARIMA | `pmdarima` + per-ticker Python loop |
| Statistical | Holt-Winters | `statsmodels` + per-ticker Python loop |
| ML Baseline | Linear Regression | PySpark MLlib |
| ML Tree | Random Forest | PySpark MLlib |
| ML Tree | GBT (Gradient Boosted Trees) | PySpark MLlib |

Pre-computed results are saved to `Dataset/results/` and read by the Streamlit dashboard.

---

## 2. Dataset

- **Source:** 30 CSV files crawled from Vietstock Finance (`Dataset/*.csv`)
- **Tickers:** All 30 VN30 constituents as of 31/08/2026
- **Date range:** 02/01/2018 → 31/08/2026
- **Target variable:** `Close_price` (univariate forecasting)
- **Frequency:** Daily trading days
- **Estimated total rows:** ~60,000 across all tickers

> Note: SSB (SeABank) was listed in March 2021, so it has fewer records than other tickers. This is expected and noted in the report.

---

## 3. Pipeline Architecture (`pipeline.py`)

### 3.1 Load & Clean

PySpark reads all 30 CSV files in one shot from the `Dataset/` folder:

```python
sdf = spark.read.option('header', 'true').csv(csv_files)
```

Comma-separated numeric strings (e.g. `"22,600"`) are cleaned with `regexp_replace`.
Only three columns are retained: `Date`, `Ticker`, `Close_price`.

---

### 3.2 Feature Engineering

All features are constructed to prevent **data leakage** — every value seen
at prediction time only uses information available strictly before that day.

| Feature | Description |
|---|---|
| `lag1`, `lag2`, `lag3`, `lag5`, `lag10` | Lagged close price (t-1 to t-10) |
| `rmean5`, `rmean10`, `rmean20` | Rolling mean ending at t-1 |
| `rstd5`, `rstd10`, `rstd20` | Rolling std ending at t-1 |
| `dow` | Day of week |
| `month` | Month of year |

---

### 3.3 Train / Test Split

- Split ratio: **80% train / 20% test**, time-ordered per ticker
- No shuffling — temporal order is always preserved
- Rows with `NaN` in any feature column are dropped before training

---

### 3.4 PySpark MLlib Models

All three ML models share the same `VectorAssembler` feature vector.
Training and prediction happen fully distributed inside Spark.

| Model | Key parameters |
|---|---|
| Linear Regression | `maxIter=200`, `regParam=0.05`, `elasticNetParam=0.0` |
| Random Forest | `numTrees=100`, `maxDepth=10`, `seed=42` |
| GBT | `maxIter=100`, `maxDepth=6`, `seed=42` |

---

### 3.5 Statistical Models (ARIMA & Holt-Winters)

Because `statsmodels` is not natively distributed, both statistical models
run in a Python loop over each ticker's Pandas DataFrame (collected from Spark).

**ARIMA:**

```python
auto_arima(train, max_p=3, max_q=3, max_d=2, stepwise=True)
```

Order `(p,d,q)` is selected automatically via AIC with ADF/KPSS stationarity tests —
not hardcoded. Falls back to the last training value if fitting fails.

**Holt-Winters:**

```python
ExponentialSmoothing(train, trend='add', seasonal=None,
                     initialization_method='estimated').fit(optimized=True)
```

Runs on raw daily data (no monthly resampling). Falls back to the last training
value if fitting fails.

Intermediate results are written to `Dataset/results/_tmp/` as Parquet,
then read back into Spark to union with the MLlib predictions.

---

### 3.6 Metrics

Each model × ticker combination is evaluated with four metrics:

| Metric | Formula |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error (%) |
| R² | Coefficient of determination |

---

### 3.7 Output Files

All results are saved to `Dataset/results/` and read directly by `app.py`:

| File | Content |
|---|---|
| `predictions.csv` | All predictions: Ticker, Date, Actual, Predicted, Model |
| `predictions.parquet` | Same as above in Parquet format |
| `metrics.csv` | Metrics per Ticker × Model: MAE, RMSE, MAPE, R² |
| `_tmp/arima_tmp.parquet` | Intermediate ARIMA output (can be deleted after run) |
| `_tmp/hw_tmp.parquet` | Intermediate Holt-Winters output (can be deleted after run) |

---

## 4. Running the Pipeline

**Environment:** Python 3.12 at `C:\venvs\bigdata\Scripts\python.exe`
(the project `.venv` is not used — Windows Long Path issues with deep paths)

**Run in PyCharm** using Scientific mode (`#%%` cells), or from terminal:

```bash
C:\venvs\bigdata\Scripts\python.exe pipeline.py
```

Expected console output when complete:

```
Loaded 61,234 rows across 30 tickers
Train: 49,012 rows | Test: 12,222 rows
Linear Regression ✔
Random Forest ✔
GBT ✔
ARIMA ✔
Holt-Winters ✔

Results saved → .../Dataset/results

=================================================================
Mean metrics across all tickers (sorted by RMSE)
=================================================================
                   MAE     RMSE     MAPE      R2
Model
Random Forest    ...      ...      ...       ...
GBT              ...      ...      ...       ...
...
=================================================================
Best model (RMSE): ...
```

---

## 5. Limitations

- **No real-time data** — pipeline runs once on static CSV files; re-crawl needed for new data.
- **ARIMA is not truly distributed** — runs per-ticker in a Python loop; slow for many tickers.
- **Holt-Winters has no seasonality** — `seasonal=None` because daily stock data has no stable seasonal pattern.
- **Tab 3 (Upload & Analyze) in the Streamlit app only supports 2 models** — Linear Trend and Holt-Winters. ARIMA, Random Forest, and GBT require PySpark and cannot run inside Streamlit Cloud. This is documented as a known limitation in the report.

---

## 6. Requirements

The pipeline requires these packages (in addition to the Streamlit dependencies in `requirements.txt`):

```
pyspark>=3.5
pmdarima>=2.0
statsmodels>=0.14
numpy>=1.26
pandas>=2.0
pyarrow>=14.0
```

Install in the `C:\venvs\bigdata` environment:

```bash
pip install pyspark pmdarima statsmodels numpy pandas pyarrow
```
