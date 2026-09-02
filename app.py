import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from pathlib import Path

st.set_page_config(page_title='Dashboard Dự báo Giá Đóng Cửa VN30', layout='wide')

ROOT       = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'Dataset' / 'results'
DATA_DIR   = ROOT / 'Dataset'
SPLIT      = 0.80

COMMA_COLS = [
    'Reference', 'Open_price', 'Close_price', 'Highest_price', 'Lowest_price',
    'Average', 'Change', 'Matched_value', 'Negotiated_value', 'Total_value', 'Market_cap'
]

MODEL_COLORS = {
    'Linear Regression': '#636EFA',
    'Random Forest':     '#EF553B',
    'GBT':               '#00CC96',
    'ARIMA':             '#AB63FA',
    'Holt-Winters':      '#FFA15A',
}

def _show_table(styler):
    st.markdown('<div style="overflow-x:auto">' + styler.to_html() + '</div>',
                unsafe_allow_html=True)

def _styled(df, min_cols=None, max_cols=None, fmt=None):
    df = df.copy().reset_index(drop=True)
    df.insert(0, 'No.', range(1, len(df) + 1))
    s = (df.style
           .hide(axis='index')
           .set_properties(**{'text-align': 'center', 'padding': '6px 12px'})
           .set_table_styles([
               {'selector': 'th', 'props': [
                   ('text-align', 'center'),
                   ('background-color', 'rgba(128,128,128,0.15)'),
                   ('font-weight', 'bold'),
                   ('padding', '8px 12px'),
               ]},
               {'selector': 'td', 'props': [
                   ('text-align', 'center'),
                   ('padding', '6px 12px'),
               ]},
           ]))
    if min_cols:
        for c in min_cols:
            if c in df.columns:
                s = s.highlight_min(subset=[c],
                                    props='background-color: #1e7e34; color: white; font-weight: bold;')
    if max_cols:
        for c in max_cols:
            if c in df.columns:
                s = s.highlight_max(subset=[c],
                                    props='background-color: #1e7e34; color: white; font-weight: bold;')
    if fmt:
        s = s.format(fmt, na_rep='—')
    return s

# ── Data loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    preds   = pd.read_csv(RESULT_DIR / 'predictions.csv', parse_dates=['Date'])
    metrics = pd.read_csv(RESULT_DIR / 'metrics.csv')
    return preds, metrics

@st.cache_data
def load_raw():
    dfs = []
    for f in DATA_DIR.glob('*.csv'):
        try:
            df = pd.read_csv(f, dtype=str)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    for col in COMMA_COLS:
        if col in raw.columns:
            raw[col] = raw[col].str.replace(',', '', regex=False)
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y', errors='coerce')
    raw = raw.dropna(subset=['Date', 'Close_price'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    return raw

# ── Tab 1 helpers ────────────────────────────────────────────────────────────

def plot_overview(df_t):
    sp = int(len(df_t) * SPLIT)
    train = df_t.iloc[:sp]
    test  = df_t.iloc[sp:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Giá đóng cửa (Close Price)', 'Lợi suất hàng ngày (Daily Return)'],
                        row_heights=[0.65, 0.35], vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=train['Date'], y=train['Close_price'],
                             name='Train', line=dict(color='steelblue', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test['Date'],  y=test['Close_price'],
                             name='Test',  line=dict(color='tomato',    width=1.2)), row=1, col=1)

    ret = df_t['Close_price'].pct_change() * 100
    colors = ['tomato' if r < 0 else 'steelblue' for r in ret]
    fig.add_trace(go.Bar(x=df_t['Date'], y=ret, name='Return (%)',
                         marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)

    fig.update_layout(height=560, margin=dict(t=50, b=70),
                      legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5))
    return fig


def plot_rolling(df_t):
    close = df_t.set_index('Date')['Close_price']
    sp    = int(len(close) * SPLIT)
    train = close.iloc[:sp]

    roll20  = train.rolling(20).mean()
    roll_std = train.rolling(20).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train,      name='Original',      line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(x=train.index, y=roll20,     name='Rolling Mean 20', line=dict(color='red',       width=2)))
    fig.add_trace(go.Scatter(x=train.index, y=roll_std,   name='Rolling Std 20',  line=dict(color='orange',    width=1.5, dash='dash')))
    fig.update_layout(height=420, margin=dict(t=90, b=60),
                      title=dict(text='Rolling Mean & Std (tập train)', y=0.97, yanchor='top'),
                      legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5))
    return fig


def plot_decompose(df_t):
    close = df_t.set_index('Date')['Close_price'].asfreq('B').ffill()
    sp    = int(len(close) * SPLIT)
    train = close.iloc[:sp]
    try:
        result = seasonal_decompose(train, model='additive', period=252)
    except Exception:
        result = seasonal_decompose(train, model='additive', period=5)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                        vertical_spacing=0.06)
    for i, (name, comp) in enumerate([
        ('Observed', result.observed), ('Trend', result.trend),
        ('Seasonal', result.seasonal), ('Residual', result.resid)
    ], 1):
        fig.add_trace(go.Scatter(x=comp.index, y=comp.values, name=name,
                                 line=dict(width=1.2)), row=i, col=1)

    fig.update_layout(height=680, showlegend=False, margin=dict(t=90, b=20),
                      title=dict(text='Phân rã chuỗi thời gian (tập train, period=252 ngày giao dịch)',
                                 y=0.98, yanchor='top'))
    return fig


def plot_acf_pacf(df_t, nlags=40):
    close = df_t.set_index('Date')['Close_price']
    sp    = int(len(close) * SPLIT)
    diff  = close.iloc[:sp].diff().dropna()

    acf_vals,  acf_ci  = acf(diff,  nlags=nlags, alpha=0.05)
    pacf_vals, pacf_ci = pacf(diff, nlags=nlags, alpha=0.05, method='ywm')
    lags = list(range(nlags + 1))

    fig = make_subplots(rows=1, cols=2, subplot_titles=['ACF (sau sai phân bậc 1)', 'PACF (sau sai phân bậc 1)'])
    for col, (vals, ci, title) in enumerate([
        (acf_vals, acf_ci, 'ACF'), (pacf_vals, pacf_ci, 'PACF')
    ], 1):
        upper = ci[:, 1] - vals
        lower = vals - ci[:, 0]
        fig.add_trace(go.Bar(x=lags, y=vals, name=title,
                             error_y=dict(type='data', array=upper, arrayminus=lower, visible=True),
                             marker_color='steelblue'), row=1, col=col)
        fig.add_hline(y=0, line_color='black', row=1, col=col)

    fig.update_layout(height=400, showlegend=False, margin=dict(t=90, b=20),
                      title=dict(text='ACF & PACF của giá đóng cửa sau sai phân bậc 1',
                                 y=0.97, yanchor='top'))
    return fig


def plot_diff(df_t):
    close = df_t.set_index('Date')['Close_price']
    sp    = int(len(close) * SPLIT)
    train = close.iloc[:sp]
    diff  = train.diff().dropna()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Dữ liệu gốc (Train)', 'Sau sai phân bậc 1'],
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=train.index, y=train.values, line=dict(color='steelblue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=diff.index,  y=diff.values,  line=dict(color='tomato',    width=1)), row=2, col=1)
    fig.update_layout(height=460, showlegend=False, margin=dict(t=90, b=20),
                      title=dict(text='Dữ liệu gốc và sau sai phân bậc 1', y=0.97, yanchor='top'))
    return fig


def adf_kpss_table(df_t):
    close = df_t.set_index('Date')['Close_price']
    sp    = int(len(close) * SPLIT)
    train = close.iloc[:sp].dropna()
    diff  = train.diff().dropna()

    rows = []
    for label, series in [('Gốc (Train)', train), ('Sai phân bậc 1', diff)]:
        adf_p  = adfuller(series, autolag='AIC')[1]
        kpss_p = kpss(series, regression='c', nlags='auto')[1]
        rows.append({'Chuỗi': label,
                     'ADF p-value': round(adf_p, 6),
                     'ADF Kết luận': 'Dừng ✔' if adf_p <= 0.05 else 'Không dừng ✘',
                     'KPSS p-value': round(kpss_p, 6),
                     'KPSS Kết luận': 'Dừng ✔' if kpss_p > 0.05 else 'Không dừng ✘'})
    return pd.DataFrame(rows)


def plot_forecast(df_t, preds_t):
    close = df_t.set_index('Date')['Close_price']
    sp    = int(len(close) * SPLIT)
    train = close.iloc[:sp]
    test  = close.iloc[sp:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train',
                             line=dict(color='lightgray', width=1.2)))
    fig.add_trace(go.Scatter(x=test.index,  y=test.values,  name='Actual (Test)',
                             line=dict(color='black', width=2)))

    for model, color in MODEL_COLORS.items():
        m_preds = preds_t[preds_t['Model'] == model].sort_values('Date')
        if m_preds.empty:
            continue
        fig.add_trace(go.Scatter(x=m_preds['Date'], y=m_preds['Predicted'],
                                 name=model, line=dict(color=color, width=1.5, dash='dot')))

    fig.update_layout(title='Dự báo so với thực tế (Test set)',
                      height=480, margin=dict(t=50, b=80),
                      legend=dict(orientation='h', yanchor='top', y=-0.12,
                                  xanchor='center', x=0.5))
    return fig


def plot_metrics_bar(metrics_t):
    colors = [MODEL_COLORS.get(m, 'gray') for m in metrics_t['Model']]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['RMSE (thấp hơn = tốt hơn)', 'MAE (thấp hơn = tốt hơn)',
                                        'MAPE % (thấp hơn = tốt hơn)', 'R² (cao hơn = tốt hơn)'],
                        vertical_spacing=0.18, horizontal_spacing=0.12)
    for (row, col), metric in zip(
        [(1,1),(1,2),(2,1),(2,2)],
        ['RMSE', 'MAE', 'MAPE', 'R2']
    ):
        fig.add_trace(go.Bar(x=metrics_t['Model'], y=metrics_t[metric],
                             marker_color=colors, name=metric,
                             text=metrics_t[metric].round(2),
                             texttemplate='%{text}', textposition='outside'),
                      row=row, col=col)
    fig.update_layout(height=600, showlegend=False, margin=dict(t=90, b=20),
                      title=dict(text='So sánh chỉ số đánh giá các mô hình', y=0.98, yanchor='top'))
    return fig

# ── Tab 2 helpers ────────────────────────────────────────────────────────────

def prepare_uploaded(df_up):
    df_up = df_up.copy()
    for col in COMMA_COLS:
        if col in df_up.columns:
            df_up[col] = df_up[col].astype(str).str.replace(',', '', regex=False)
            df_up[col] = pd.to_numeric(df_up[col], errors='coerce')
    if 'Date' in df_up.columns:
        df_up['Date'] = pd.to_datetime(df_up['Date'], dayfirst=True, errors='coerce')
    df_up = (df_up.dropna(subset=['Close_price', 'Date'])
                  .sort_values('Date')
                  .reset_index(drop=True))
    return df_up


def forecast_uploaded(df_clean):
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    sp         = int(len(df_clean) * SPLIT)
    train_df   = df_clean.iloc[:sp]
    test_df    = df_clean.iloc[sp:]
    train_vals = train_df['Close_price'].values.astype(float)
    test_vals  = test_df['Close_price'].values.astype(float)

    # Linear Trend
    X_tr    = np.arange(sp).reshape(-1, 1)
    lr      = LinearRegression().fit(X_tr, train_vals)
    X_te    = np.arange(sp, sp + len(test_vals)).reshape(-1, 1)
    lr_pred = lr.predict(X_te)

    # Holt-Winters (trend only, no seasonal)
    try:
        hw_pred = (ExponentialSmoothing(
            train_vals, trend='add', seasonal=None,
            initialization_method='estimated'
        ).fit(optimized=True).forecast(len(test_vals)))
    except Exception:
        hw_pred = np.full(len(test_vals), float(train_vals[-1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df['Date'], y=train_vals, name='Train',
                             line=dict(color='steelblue', width=1.2)))
    fig.add_trace(go.Scatter(x=test_df['Date'],  y=test_vals,  name='Actual (Test)',
                             line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=test_df['Date'],  y=lr_pred, name='Linear Trend',
                             line=dict(color='tomato', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=test_df['Date'],  y=hw_pred, name='Holt-Winters',
                             line=dict(color='#FFA15A', width=1.5, dash='dash')))
    fig.update_layout(height=480, margin=dict(t=90, b=80),
                      title=dict(text='Dự báo so với thực tế (Test set)', y=0.97, yanchor='top'),
                      legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5))

    def _metrics(actual, pred, name):
        y, yh = actual, np.array(pred).astype(float)
        mask  = y != 0
        mae   = float(np.mean(np.abs(y - yh)))
        rmse  = float(np.sqrt(np.mean((y - yh) ** 2)))
        mape  = float(np.mean(np.abs((y[mask] - yh[mask]) / y[mask])) * 100) if mask.any() else None
        ss_r  = np.sum((y - yh) ** 2)
        ss_t  = np.sum((y - y.mean()) ** 2)
        r2    = float(1 - ss_r / ss_t) if ss_t else 0.0
        return {'Mô hình': name,
                'MAE':     round(mae,  2),
                'RMSE':    round(rmse, 2),
                'MAPE (%)': round(mape, 2) if mape is not None else None,
                'R²':      round(r2,   4)}

    metrics_df = pd.DataFrame([
        _metrics(test_vals, lr_pred, 'Linear Trend'),
        _metrics(test_vals, hw_pred, 'Holt-Winters'),
    ])
    return fig, metrics_df

# ── Main layout ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
table td, table th { text-align: center !important; padding: 6px 12px !important; }
table th { background-color: rgba(128,128,128,0.15) !important; font-weight: bold !important; }
table { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

st.title('Dashboard Dự báo Giá Đóng Cửa VN30 — Phân tích Chuỗi Thời gian với 5 Mô hình')
st.markdown("""
<p style="color:#888; font-size:14px;">
<abbr title="Lê Nhật Thanh — C25611254&#10;Lê Ngọc Phú — C25611253&#10;Huỳnh Trúc Ngân — C25611251&#10;Phạm Thị Thúy Kiều — C25611258&#10;Trịnh Quang Tân — C25611256"
      style="cursor:help; text-decoration: underline dotted; color:#888; font-size:14px;">Nhóm Gia đình Haha</abbr>
&nbsp;|&nbsp; GVHD: TS. Nguyễn Thôn Dã &nbsp;
</p>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Phân tích theo Ticker', 'So sánh 30 Tickers', 'Upload & Analyze'])

# ════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════
with tab1:
    if not (RESULT_DIR / 'predictions.csv').exists():
        st.error('Chưa có file kết quả. Chạy pipeline.py trước.')
        st.stop()

    preds, metrics = load_results()

    with st.spinner('Đang tải dữ liệu gốc (30 CSV)...'):
        raw = load_raw()

    tickers = sorted(preds['Ticker'].unique())
    col_sel, col_info = st.columns([2, 5])
    with col_sel:
        ticker = st.selectbox('Chọn Ticker:', tickers, index=0)
    with col_info:
        st.metric('Tổng số tickers', len(tickers))

    preds_t   = preds[preds['Ticker'] == ticker].copy()
    metrics_t = metrics[metrics['Ticker'] == ticker].copy()

    if not raw.empty and ticker in raw['Ticker'].values:
        df_t = raw[raw['Ticker'] == ticker].reset_index(drop=True)
    else:
        df_t = None
        if raw.empty:
            st.warning(f'load_raw() trả về empty DataFrame. Kiểm tra đường dẫn: {DATA_DIR}')
        else:
            st.warning(f'Ticker "{ticker}" không tìm thấy trong raw data. Các tickers có sẵn: {sorted(raw["Ticker"].unique())[:5]}')

    # ── Section 1: Overview ──────────────────────────────────
    st.subheader('1. Tổng quan chuỗi thời gian & phân chia Train/Test')
    if df_t is not None:
        st.plotly_chart(plot_overview(df_t), use_container_width=True)
        sp = int(len(df_t) * SPLIT)
        c1, c2, c3 = st.columns(3)
        c1.metric('Tổng số ngày', len(df_t))
        c2.metric('Train', sp)
        c3.metric('Test', len(df_t) - sp)

        with st.expander('Thống kê mô tả Close_price'):
            _desc = df_t['Close_price'].describe().to_frame().T
            _show_table(_styled(_desc, fmt={c: '{:,.2f}' for c in _desc.columns}))
    else:
        st.info('Không tìm thấy file CSV gốc để hiện biểu đồ tổng quan.')

    st.divider()
    # ── Section 2: Rolling stats ─────────────────────────────
    st.subheader('2. Rolling Mean & Rolling Std (tập train)')
    if df_t is not None:
        st.plotly_chart(plot_rolling(df_t), use_container_width=True)

    st.divider()
    # ── Section 3: Stationarity ──────────────────────────────
    st.subheader('3. Kiểm định tính dừng (ADF & KPSS)')
    if df_t is not None:
        st.plotly_chart(plot_diff(df_t), use_container_width=True)
        with st.spinner('Đang chạy ADF & KPSS...'):
            stat_df = adf_kpss_table(df_t)
        _show_table(_styled(stat_df,
                         fmt={'ADF p-value': '{:.6f}', 'KPSS p-value': '{:.6f}'}))

    st.divider()
    # ── Section 4: ACF/PACF ──────────────────────────────────
    st.subheader('4. ACF & PACF (sau sai phân bậc 1)')
    if df_t is not None:
        st.plotly_chart(plot_acf_pacf(df_t), use_container_width=True)

    st.divider()
    # ── Section 5: Decomposition ─────────────────────────────
    st.subheader('5. Phân rã chuỗi thời gian (Seasonal Decomposition)')
    if df_t is not None:
        with st.spinner('Đang phân rã...'):
            fig_dec = plot_decompose(df_t)
        st.plotly_chart(fig_dec, use_container_width=True)

    st.divider()
    # ── Section 6: Forecast ──────────────────────────────────
    st.subheader('6. Kết quả dự báo — 5 mô hình')
    if df_t is not None:
        st.plotly_chart(plot_forecast(df_t, preds_t), use_container_width=True)
    else:
        # Fallback: only test period
        fig_f = go.Figure()
        for model, color in MODEL_COLORS.items():
            m = preds_t[preds_t['Model'] == model].sort_values('Date')
            if m.empty:
                continue
            fig_f.add_trace(go.Scatter(x=m['Date'], y=m['Actual'], name='Actual',
                                       line=dict(color='black', width=2), showlegend=(model == list(MODEL_COLORS)[0])))
            fig_f.add_trace(go.Scatter(x=m['Date'], y=m['Predicted'], name=model,
                                       line=dict(color=color, width=1.5, dash='dot')))
        fig_f.update_layout(height=420, title='Dự báo vs Thực tế (Test set)')
        st.plotly_chart(fig_f, use_container_width=True)

    st.divider()
    # ── Section 7: Metrics ───────────────────────────────────
    st.subheader('7. Đánh giá mô hình — Metrics')
    if not metrics_t.empty:
        st.plotly_chart(plot_metrics_bar(metrics_t), use_container_width=True)
        _mt = (metrics_t[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']]
               .sort_values('RMSE').reset_index(drop=True)
               .rename(columns={'R2': 'R²', 'MAPE': 'MAPE (%)'}))
        _show_table(_styled(_mt,
                         min_cols=['MAE', 'RMSE', 'MAPE (%)'],
                         max_cols=['R²'],
                         fmt={'MAE': '{:,.2f}', 'RMSE': '{:,.2f}',
                              'MAPE (%)': '{:.2f}', 'R²': '{:.4f}'}))

# ════════════════════════════════════════════════════════════
# TAB 2 — So sánh 30 Tickers
# ════════════════════════════════════════════════════════════
with tab2:
    if not (RESULT_DIR / 'metrics.csv').exists():
        st.error('Chưa có file kết quả. Chạy pipeline.py trước.')
        st.stop()

    preds2, metrics2 = load_results()

    # ── Section 1: Tổng hợp metrics ──────────────────────────
    st.subheader('1. Metrics trung bình theo mô hình (30 tickers)')
    summary = (metrics2.groupby('Model')[['MAE', 'RMSE', 'MAPE', 'R2']]
               .mean().round(4).sort_values('RMSE').reset_index()
               .rename(columns={'R2': 'R²', 'MAPE': 'MAPE (%)'}))
    _show_table(_styled(summary,
                     min_cols=['MAE', 'RMSE', 'MAPE (%)'],
                     max_cols=['R²'],
                     fmt={'MAE': '{:,.2f}', 'RMSE': '{:,.2f}',
                          'MAPE (%)': '{:.2f}', 'R²': '{:.4f}'}))

    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(summary, x='Model', y='RMSE', color='Model',
                          color_discrete_map=MODEL_COLORS,
                          title='RMSE trung bình (30 tickers)', text_auto='.0f')
        fig_rmse.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
    with c2:
        fig_mae = px.bar(summary, x='Model', y='MAE', color='Model',
                         color_discrete_map=MODEL_COLORS,
                         title='MAE trung bình (30 tickers)', text_auto='.0f')
        fig_mae.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_mape = px.bar(summary, x='Model', y='MAPE (%)', color='Model',
                          color_discrete_map=MODEL_COLORS,
                          title='MAPE % trung bình (30 tickers)', text_auto='.2f')
        fig_mape.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_mape, use_container_width=True)
    with c4:
        fig_r2 = px.bar(summary, x='Model', y='R²', color='Model',
                        color_discrete_map=MODEL_COLORS,
                        title='R² trung bình (30 tickers)', text_auto='.4f')
        fig_r2.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()
    # ── Section 2: Heatmap (4 chỉ số) ────────────────────────
    st.subheader('2. Heatmap Ticker × Mô hình')

    model_order  = summary['Model'].tolist()
    ticker_order = sorted(metrics2['Ticker'].unique().tolist())

    heatmap_cfg = [
        ('RMSE',     'RdYlGn_r', 'Màu đỏ = RMSE cao (kém), xanh = RMSE thấp (tốt).',     '{:.0f}'),
        ('MAE',      'RdYlGn_r', 'Màu đỏ = MAE cao (kém), xanh = MAE thấp (tốt).',       '{:.0f}'),
        ('MAPE',     'RdYlGn_r', 'Màu đỏ = MAPE cao (kém), xanh = MAPE thấp (tốt).',     '{:.2f}'),
        ('R2',       'RdYlGn',   'Màu xanh = R² cao (tốt), đỏ = R² thấp (kém).',         '{:.4f}'),
    ]

    for metric, colorscale, caption, fmt in heatmap_cfg:
        st.markdown(f'**{metric}** — {caption}')
        pivot = (metrics2.pivot_table(index='Ticker', columns='Model', values=metric)
                         .reindex(columns=model_order)
                         .reindex(ticker_order))
        text_vals = np.vectorize(lambda v: fmt.format(v) if not np.isnan(v) else '')(pivot.values)
        label = 'R²' if metric == 'R2' else metric
        fig_h = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            text=text_vals,
            texttemplate='%{text}',
            textfont=dict(size=9),
            colorbar=dict(title=label),
            hovertemplate=f'Ticker: %{{y}}<br>Model: %{{x}}<br>{label}: %{{z:.4f}}<extra></extra>',
        ))
        fig_h.update_layout(
            title=f'{label} Heatmap — 30 Tickers × 5 Mô hình',
            height=700, margin=dict(t=60, b=40, l=60, r=20),
            xaxis_title='Mô hình', yaxis_title='Ticker',
        )
        st.plotly_chart(fig_h, use_container_width=True)

    st.divider()
    # ── Section 3: Best model per ticker ─────────────────────
    st.subheader('3. Mô hình tốt nhất cho từng Ticker (theo RMSE)')

    _best_raw = (metrics2.sort_values('RMSE')
                         .groupby('Ticker', sort=True)
                         .first()
                         .reset_index()[['Ticker', 'Model', 'RMSE', 'R2', 'MAE', 'MAPE']])
    best = pd.DataFrame({
        'Ticker':     _best_raw['Ticker'],
        'Best Model': _best_raw['Model'],
        'RMSE':       _best_raw['RMSE'].round(2),
        'R²':         _best_raw['R2'].round(4),
        'MAE':        _best_raw['MAE'].round(2),
        'MAPE (%)':   _best_raw['MAPE'].round(2),
    })
    _show_table(_styled(best,
                     min_cols=['RMSE', 'MAE', 'MAPE (%)'],
                     max_cols=['R²'],
                     fmt={'RMSE': '{:,.2f}', 'MAE': '{:,.2f}',
                          'MAPE (%)': '{:.2f}', 'R²': '{:.4f}'}))

    win_counts = best['Best Model'].value_counts().reset_index()
    win_counts.columns = ['Model', 'Số ticker thắng']
    fig_win = px.bar(
        win_counts, x='Model', y='Số ticker thắng', color='Model',
        color_discrete_map=MODEL_COLORS,
        title='Phân phối mô hình tốt nhất trên 30 tickers (theo RMSE thấp nhất)',
        text='Số ticker thắng',
    )
    fig_win.update_traces(textposition='outside')
    fig_win.update_layout(height=380, showlegend=False,
                          yaxis=dict(range=[0, win_counts['Số ticker thắng'].max() + 2]))
    st.plotly_chart(fig_win, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Upload & Analyze
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader('Upload file CSV để phân tích nhanh')
    st.caption('File cần có cột Date (dd/MM/yyyy) và Close_price. Không dùng PySpark — chạy trực tiếp bằng pandas.')

    uploaded = st.file_uploader('Chọn file CSV', type='csv')
    if uploaded:
        df_raw_up = pd.read_csv(uploaded, dtype=str)
        st.write(f'Đã tải: **{uploaded.name}** — {len(df_raw_up):,} dòng, {df_raw_up.shape[1]} cột')

        try:
            df_up_clean = prepare_uploaded(df_raw_up)
        except Exception as e:
            st.error(f'Lỗi xử lý dữ liệu: {e}')
            st.stop()

        if df_up_clean.empty or len(df_up_clean) < 20:
            st.error('Không đủ dữ liệu để phân tích (cần ít nhất 20 dòng hợp lệ).')
            st.stop()

        sp_up = int(len(df_up_clean) * SPLIT)

        # A: Thống kê mô tả
        st.subheader('A. Thống kê mô tả')
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric('Tổng số ngày giao dịch', len(df_up_clean))
        ca2.metric('Train (80%)', sp_up)
        ca3.metric('Test (20%)',  len(df_up_clean) - sp_up)
        _desc_up = df_up_clean['Close_price'].describe().to_frame().T.round(2)
        _show_table(_styled(_desc_up, fmt={c: '{:,.2f}' for c in _desc_up.columns}))

        st.divider()
        # B: Tổng quan
        st.subheader('B. Tổng quan chuỗi thời gian & phân chia Train/Test')
        st.plotly_chart(plot_overview(df_up_clean), use_container_width=True)

        st.divider()
        # C: Rolling stats
        st.subheader('C. Rolling Mean & Std (tập train)')
        st.plotly_chart(plot_rolling(df_up_clean), use_container_width=True)

        st.divider()
        # D: Kiểm định tính dừng
        st.subheader('D. Kiểm định tính dừng (ADF & KPSS)')
        st.plotly_chart(plot_diff(df_up_clean), use_container_width=True)
        with st.spinner('Đang chạy ADF & KPSS...'):
            stat_df_up = adf_kpss_table(df_up_clean)
        _show_table(_styled(stat_df_up,
                         fmt={'ADF p-value': '{:.6f}', 'KPSS p-value': '{:.6f}'}))

        st.divider()
        # E: ACF / PACF
        st.subheader('E. ACF & PACF (sau sai phân bậc 1)')
        nlags_up = min(40, sp_up // 3)
        st.plotly_chart(plot_acf_pacf(df_up_clean, nlags=nlags_up), use_container_width=True)

        st.divider()
        # F: Dự báo
        st.subheader('F. Dự báo — Linear Trend & Holt-Winters')
        with st.spinner('Đang chạy mô hình dự báo...'):
            try:
                fig_up, metrics_up = forecast_uploaded(df_up_clean)
            except Exception as e:
                st.error(f'Lỗi dự báo: {e}')
                st.stop()

        st.plotly_chart(fig_up, use_container_width=True)
        _mu = metrics_up.sort_values('RMSE').reset_index(drop=True)
        _show_table(_styled(_mu,
                         min_cols=['MAE', 'RMSE', 'MAPE (%)'],
                         max_cols=['R²'],
                         fmt={'MAE': '{:,.2f}', 'RMSE': '{:,.2f}',
                              'MAPE (%)': '{:.2f}', 'R²': '{:.4f}'}))
        st.caption('Linear Trend: hồi quy tuyến tính theo index thời gian | Holt-Winters: làm trơn mũ có xu hướng (additive).')
