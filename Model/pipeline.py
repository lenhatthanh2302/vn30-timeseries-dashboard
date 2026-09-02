#%% - Libraries
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (StructType, StructField, StringType, DateType, DoubleType)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (LinearRegression as SparkLR,
                                   RandomForestRegressor as SparkRF,
                                   GBTRegressor as SparkGBT)

from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss

#%% - Config
TARGET    = 'Close_price'
SPLIT     = 0.80
SEED      = 42
LAG_DAYS  = [1, 2, 3, 5, 10]
ROLL_WINS = [5, 10, 20]

COMMA_COLS = [
    'Reference', 'Open_price', 'Close_price', 'Highest_price', 'Lowest_price',
    'Average', 'Change', 'Matched_value', 'Negotiated_value', 'Total_value', 'Market_cap'
]

#%% - Paths
try:
    root = Path(__file__).resolve().parents[1]
except NameError:
    _cwd = Path.cwd()
    root = _cwd if (_cwd / 'Dataset').exists() else _cwd.parent

dataset_dir = root / 'Dataset'
result_dir  = dataset_dir / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

#%% - Spark Session
import os
_py = r'C:\Users\ThanhT\AppData\Local\Programs\Python\Python312\python.exe'
os.environ['PYSPARK_PYTHON']        = _py
os.environ['PYSPARK_DRIVER_PYTHON'] = _py

spark = (SparkSession.builder
         .appName('VN30_TimeSeries_Pipeline')
         .config('spark.driver.memory', '6g')
         .config('spark.sql.shuffle.partitions', '8')
         .getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

#%% - Load & Clean all CSVs in Dataset/
csv_files = [str(p) for p in dataset_dir.glob('*.csv')]
sdf = (spark.read
       .option('header', 'true')
       .option('inferSchema', 'false')
       .csv(csv_files))

# Xóa dấu phẩy trong số (vd: "22,600" → 22600)
for col in COMMA_COLS:
    if col in sdf.columns:
        sdf = sdf.withColumn(col, F.regexp_replace(col, ',', '').cast('double'))

sdf = (sdf
       .withColumn('Date', F.to_date('Date', 'dd/MM/yyyy'))
       .select('Date', 'Ticker', TARGET)
       .filter(F.col(TARGET).isNotNull())
       .orderBy('Ticker', 'Date'))

n_rows    = sdf.count()
n_tickers = sdf.select('Ticker').distinct().count()
print(f'\nLoaded {n_rows:,} rows across {n_tickers} tickers')

#%% - Feature Engineering (univariate — chỉ dùng Close_price)
w_ticker = Window.partitionBy('Ticker').orderBy('Date')

# Đánh số hàng per ticker để tách train/test theo thời gian
sdf = sdf.withColumn('_rn', F.row_number().over(w_ticker))
cnt = sdf.groupBy('Ticker').agg(F.max('_rn').alias('_n'))
sdf = sdf.join(cnt, on='Ticker')
sdf = sdf.withColumn('split',
    F.when(F.col('_rn') <= F.col('_n') * SPLIT, 'train').otherwise('test'))

# Lagged close price
for lag in LAG_DAYS:
    sdf = sdf.withColumn(f'lag{lag}', F.lag(TARGET, lag).over(w_ticker))

# Rolling mean & std (kết thúc tại t-1 để tránh data leakage)
for rw in ROLL_WINS:
    wb = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-rw, -1)
    sdf = sdf.withColumn(f'rmean{rw}', F.mean(TARGET).over(wb))
    sdf = sdf.withColumn(f'rstd{rw}',  F.stddev(TARGET).over(wb))

sdf = sdf.withColumn('dow',   F.dayofweek('Date'))
sdf = sdf.withColumn('month', F.month('Date'))

feat_cols = (
    [f'lag{i}' for i in LAG_DAYS] +
    [f'rmean{rw}' for rw in ROLL_WINS] +
    [f'rstd{rw}'  for rw in ROLL_WINS] +
    ['dow', 'month']
)

sdf_ml = sdf.dropna(subset=feat_cols + [TARGET]).cache()
train_sdf = sdf_ml.filter(F.col('split') == 'train')
test_sdf  = sdf_ml.filter(F.col('split') == 'test')

print(f'Train: {train_sdf.count():,} rows | Test: {test_sdf.count():,} rows')

#%% - PySpark MLlib Models (Linear Regression, Random Forest, GBT)
assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')
train_vec = assembler.transform(train_sdf).cache()
test_vec  = assembler.transform(test_sdf)

pred_cols = ['Ticker', 'Date', 'Actual', 'Predicted', 'Model']
all_preds = []

def _collect(spark_preds, name):
    return spark_preds.select(
        'Ticker', 'Date',
        F.col(TARGET).alias('Actual'),
        F.col('prediction').alias('Predicted'),
        F.lit(name).alias('Model')
    )

lr_fit  = SparkLR(featuresCol='features', labelCol=TARGET,
                   maxIter=200, regParam=0.05, elasticNetParam=0.0).fit(train_vec)
all_preds.append(_collect(lr_fit.transform(test_vec), 'Linear Regression'))
print('Linear Regression ✔')

rf_fit  = SparkRF(featuresCol='features', labelCol=TARGET,
                   numTrees=100, maxDepth=10, seed=SEED).fit(train_vec)
all_preds.append(_collect(rf_fit.transform(test_vec), 'Random Forest'))
print('Random Forest ✔')

gbt_fit = SparkGBT(featuresCol='features', labelCol=TARGET,
                    maxIter=100, maxDepth=6, seed=SEED).fit(train_vec)
all_preds.append(_collect(gbt_fit.transform(test_vec), 'GBT'))
print('GBT ✔')

#%% - Statistical Models via applyInPandas (chạy statsmodels per ticker)
stat_schema = StructType([
    StructField('Ticker',    StringType()),
    StructField('Date',      StringType()),
    StructField('Actual',    DoubleType()),
    StructField('Predicted', DoubleType()),
    StructField('Model',     StringType()),
])

def _check_stationary(series):
    adf_p  = adfuller(series.dropna(), autolag='AIC')[1]
    kpss_p = kpss(series.dropna(), regression='c', nlags='auto')[1]
    return (adf_p <= 0.05) and (kpss_p > 0.05)

def _arima(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf  = pdf.sort_values('Date').reset_index(drop=True)
    vals = pdf[TARGET].astype(float).values
    sp   = int(len(vals) * SPLIT)
    train, test = vals[:sp], vals[sp:]
    dates = pdf['Date'].iloc[sp:].values

    try:
        fit  = auto_arima(
            train, max_p=3, max_q=3, max_d=2,
            stepwise=True, suppress_warnings=True, error_action='ignore'
        )
        pred = fit.predict(n_periods=len(test))
    except Exception:
        pred = np.full(len(test), float(train[-1]))

    return pd.DataFrame({
        'Ticker':    pdf['Ticker'].iloc[0],
        'Date':      [str(d)[:10] for d in pd.to_datetime(dates)],
        'Actual':    test.astype(float),
        'Predicted': np.array(pred).astype(float),
        'Model':     'ARIMA',
    })

def _hw(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf  = pdf.sort_values('Date').reset_index(drop=True)
    vals = pdf[TARGET].astype(float).values
    sp   = int(len(vals) * SPLIT)
    train, test = vals[:sp], vals[sp:]
    dates = pdf['Date'].iloc[sp:].values

    # Không resample monthly — giữ nguyên daily data
    try:
        fit  = ExponentialSmoothing(
            train, trend='add', seasonal=None,
            initialization_method='estimated'
        ).fit(optimized=True)
        pred = fit.forecast(len(test))
    except Exception:
        pred = np.full(len(test), float(train[-1]))

    return pd.DataFrame({
        'Ticker':    pdf['Ticker'].iloc[0],
        'Date':      [str(d)[:10] for d in pd.to_datetime(dates)],
        'Actual':    test.astype(float),
        'Predicted': np.array(pred).astype(float),
        'Model':     'Holt-Winters',
    })

sdf_stat = sdf.select('Date', 'Ticker', TARGET).dropna()
sdf_stat_pd = sdf_stat.orderBy('Ticker', 'Date').toPandas()

arima_rows, hw_rows = [], []
for ticker, pdf in sdf_stat_pd.groupby('Ticker', sort=True):
    arima_rows.append(_arima(pdf.copy()))
    hw_rows.append(_hw(pdf.copy()))

arima_pd = pd.concat(arima_rows, ignore_index=True)
hw_pd    = pd.concat(hw_rows,    ignore_index=True)
print('ARIMA ✔')
print('Holt-Winters ✔')

_tmp_dir = result_dir / '_tmp'
_tmp_dir.mkdir(exist_ok=True)
arima_pd.to_parquet(str(_tmp_dir / 'arima_tmp.parquet'), index=False)
hw_pd.to_parquet(str(_tmp_dir / 'hw_tmp.parquet'),    index=False)

arima_sdf = (spark.read.parquet(str(_tmp_dir / 'arima_tmp.parquet'))
             .withColumn('Date', F.to_date('Date', 'yyyy-MM-dd')))
hw_sdf    = (spark.read.parquet(str(_tmp_dir / 'hw_tmp.parquet'))
             .withColumn('Date', F.to_date('Date', 'yyyy-MM-dd')))

all_preds.append(arima_sdf)
all_preds.append(hw_sdf)

#%% - Union predictions from all 5 models
preds = all_preds[0].select(pred_cols)
for p in all_preds[1:]:
    preds = preds.union(p.select(pred_cols))

#%% - Metrics (per Ticker, per Model)
metrics_schema = StructType([
    StructField('Ticker', StringType()),
    StructField('Model',  StringType()),
    StructField('MAE',    DoubleType()),
    StructField('RMSE',   DoubleType()),
    StructField('MAPE',   DoubleType()),
    StructField('R2',     DoubleType()),
])

def _metrics(pdf: pd.DataFrame) -> pd.DataFrame:
    y   = pdf['Actual'].values.astype(float)
    yh  = pdf['Predicted'].values.astype(float)
    mask = y != 0

    mae  = float(np.mean(np.abs(y - yh)))
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    mape = float(np.mean(np.abs((y[mask] - yh[mask]) / y[mask])) * 100) if mask.any() else np.nan
    ss_r = np.sum((y - yh) ** 2)
    ss_t = np.sum((y - y.mean()) ** 2)
    r2   = float(1 - ss_r / ss_t) if ss_t else 0.0

    return pd.DataFrame({
        'Ticker': [pdf['Ticker'].iloc[0]],
        'Model':  [pdf['Model'].iloc[0]],
        'MAE':    [round(mae,  4)],
        'RMSE':   [round(rmse, 4)],
        'MAPE':   [round(mape, 4) if not np.isnan(mape) else None],
        'R2':     [round(r2,   4)],
    })

preds_pd = preds.orderBy('Model', 'Ticker', 'Date').toPandas()
metrics_rows = []
for (ticker, model), grp in preds_pd.groupby(['Ticker', 'Model'], sort=True):
    y    = grp['Actual'].values.astype(float)
    yh   = grp['Predicted'].values.astype(float)
    mask = y != 0
    mae  = float(np.mean(np.abs(y - yh)))
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    mape = float(np.mean(np.abs((y[mask] - yh[mask]) / y[mask])) * 100) if mask.any() else None
    ss_r = np.sum((y - yh) ** 2)
    ss_t = np.sum((y - y.mean()) ** 2)
    r2   = float(1 - ss_r / ss_t) if ss_t else 0.0
    metrics_rows.append({
        'Ticker': ticker, 'Model': model,
        'MAE':  round(mae,  4), 'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4) if mape is not None else None,
        'R2':   round(r2,   4),
    })
metrics_pd = pd.DataFrame(metrics_rows)

#%% - Save results (Streamlit đọc từ đây)

preds_pd.to_csv(result_dir / 'predictions.csv',      index=False)
preds_pd.to_parquet(result_dir / 'predictions.parquet', index=False)
metrics_pd.to_csv(result_dir / 'metrics.csv',        index=False)

print(f'\nResults saved → {result_dir}')

#%% - Summary
summary = (metrics_pd
           .groupby('Model')[['MAE', 'RMSE', 'MAPE', 'R2']]
           .mean()
           .round(4)
           .sort_values('RMSE'))

print('\n' + '='*65)
print('Mean metrics across all tickers (sorted by RMSE)')
print('='*65)
print(summary.to_string())
print(f"\nBest model (RMSE): {summary['RMSE'].idxmin()}")
print('='*65)

spark.stop()
