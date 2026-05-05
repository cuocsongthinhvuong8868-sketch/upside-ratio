import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.ar_model import AutoReg

# Thư viện cho AI
import google.generativeai as genai

# Cấu hình API VNSTOCK
os.environ['VNSTOCK_API_KEY'] = 'vnstock_17b56a86b930db526e25e8de447a0bfd'
from vnstock import Quote

st.set_page_config(page_title="Hybrid MC Breadth Model", page_icon="🧬", layout="wide")

# ==========================================
# 1. CẤU HÌNH & HÀM QUẢN LÝ DỮ LIỆU
# ==========================================
CSV_FILE = 'vn_prices_cache.csv'
VNINDEX_FILE = 'vnindex_cache.csv'

DEFAULT_SYMBOLS = [
'SHB', 'NVL', 'BSR', 'TCB', 'VRE', 'VIX', 'CII', 'PVT', 'TCH', 'HPG', 
    'MBB', 'SSI', 'VCG', 'EIB', 'FPT', 'VHM', 'DXG', 'STB', 'HDB', 'VPB', 
    'VCB', 'ACB', 'HCM', 'POW', 'CTG', 'BID', 'PLX', 'GEX', 'PVD', 'DIG', 
    'DPM', 'EVF', 'MWG', 'VIB', 'VIC', 'VND', 'VPI', 'PDR', 'VCI', 'TPB', 
    'KHG', 'MSB', 'VSC', 'MSN', 'DCM', 'HHV', 'HSG', 'GEL', 'VNM', 'HAG', 
    'NKG', 'CRC', 'HHS', 'HDG', 'PVP', 'GVR', 'DGC', 'VCK', 'DGW', 'GAS', 
    'HDC', 'DBC', 'REE', 'KBC', 'KDH', 'VNE', 'PNJ', 'VHC', 'NT2', 'NAB', 
    'SAB', 'SSB', 'PC1', 'DXS', 'VJC', 'TCX', 'LCG', 'VPL', 'DLG', 'OCB', 
    'BAF', 'SCR', 'NLG', 'IDI', 'LPB', 'CTD', 'IJC', 'HQC', 'VOS', 'CTI', 
    'GMD', 'HHP', 'TDH', 'KDC', 'NTL', 'DPG', 'ORS', 'APG', 'GEG', 'YEG', 
    'HAH', 'PAC', 'LDG', 'GEE', 'BVH', 'DCL', 'SZC', 'BWE', 'HTN', 'FCN', 
    'NAF', 'TTA', 'KSB', 'FIR', 'AAA', 'TDP', 'KOS', 'TV2', 'VPX', 'EVG', 
    'FRT', 'TCO', 'SBT', 'DPR', 'TCM', 'PET', 'TTF', 'ANV', 'HVN', 'VDS', 
    'DSE', 'E1VFVN30', 'CTS', 'CTF', 'VAB', 'TLD', 'HCD', 'DRH', 'FTS', 'ABS', 
    'VGC', 'PAN', 'VTP', 'TCI', 'SHI', 'DRC', 'OGC', 'HID', 'QCG', 'KLB', 
    'VPG', 'ELC', 'LGL', 'HT1', 'BCM', 'HPX', 'GIL', 'TSA', 'SMC', 'BMI', 
    'PLP', 'ASP', 'HSL', 'ITC', 'MSH', 'GSP', 'CMG', 'PTL', 'FIT', 'TNH', 
    'TDG', 'CDC', 'ASM', 'HAX', 'VPH', 'DC4', 'CTR', 'VTO', 'AGG', 'VIP', 
    'HVH', 'TDC', 'C32', 'BSI', 'TLH', 'AGR', 'HAR', 'CSV', 'ST8', 'DHC', 
    'VVS', 'SIP', 'SCS', 'PPC', 'MIG', 'PTB', 'BFC', 'APH', 'RYG', 'FUEVFVND', 
    'BMP', 'LSS', 'RAL', 'SKG', 'NO1', 'CRE', 'SBG', 'DAH', 'MCM', 'PHR', 
    'HII', 'HPA', 'TAL', 'FUETCC50', 'MCH', 'PGC', 'LHG', 'TSC', 'SGR', 'SAM', 
    'BIC', 'CSM', 'NHA', 'DQC', 'TYA', 'TLG', 'CCL', 'SJS', 'SJD', 'JVC', 
    'AFX', 'ANT', 'BKG', 'TVS', 'VRC', 'CMX', 'TNI', 'D2D', 'FMC', 'VNS', 
    'C47', 'TRC', 'CKG', 'DHA', 'NNC', 'TIP', 'FUEVN100', 'STK', 'MCP', 'KMR'
    
]

@st.cache_data(ttl=3600)
def load_cached_prices():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE, index_col=0, parse_dates=True)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_cached_vnindex():
    if os.path.exists(VNINDEX_FILE):
        return pd.read_csv(VNINDEX_FILE, index_col=0, parse_dates=True)
    return pd.DataFrame()

def fetch_prices_kbs(symbols, start_date, end_date):
    all_prices = pd.DataFrame()
    my_bar = st.progress(0, text=f"Đang tải dữ liệu Cổ phiếu từ {start_date} đến {end_date}...")
    total = len(symbols)
    for i, symbol in enumerate(symbols):
        try:
            quote = Quote(symbol=symbol, source='KBS')
            df = quote.history(start=start_date, end=end_date, interval='1D')
            if df is not None and not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()
                df.set_index('time', inplace=True)
                all_prices[symbol] = df['close']
            time.sleep(1.1)
        except Exception:
            pass
        my_bar.progress((i + 1) / total, text=f"Đang xử lý {symbol} ({i+1}/{total})")
    my_bar.empty()
    return all_prices

def fetch_vnindex_data(start_date, end_date):
    for source in ['KBS']:
        try:
            quote = Quote(symbol='VNINDEX', source=source)
            df = quote.history(start=start_date, end=end_date, interval='1D')
            if df is not None and not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()
                df.set_index('time', inplace=True)
                return df[['close']].rename(columns={'close': 'VNINDEX'})
        except Exception:
            continue
    return pd.DataFrame()

# ==========================================
# 2. HYBRID ENSEMBLE ENGINE (CORE)
# ==========================================
def run_hybrid_ensemble_mc(raw_ratio_series, days_to_sim=20, num_sims=3000):
    """Hàm lõi chạy chung cho cả Upside và Downside để tránh lặp code"""
    p_raw = np.clip(raw_ratio_series.values, 0.1, 99.9) / 100.0
    last_p = p_raw[-1]
    
    # --- ĐỘNG CƠ 1: LOGIT BOOTSTRAP ---
    y = np.log(p_raw / (1 - p_raw))
    model_ar = AutoReg(y, lags=1, old_names=False).fit()
    c_emp = model_ar.params[0]
    phi_emp = model_ar.params[1]
    resid_emp = model_ar.resid
    
    sim_y = np.zeros((days_to_sim, num_sims))
    sim_y[0, :] = c_emp + phi_emp * y[-1] + np.random.choice(resid_emp, size=num_sims, replace=True)
    for t in range(1, days_to_sim):
        sim_y[t, :] = c_emp + phi_emp * sim_y[t-1, :] + np.random.choice(resid_emp, size=num_sims, replace=True)
    sim_p_emp = (1 / (1 + np.exp(-sim_y))) 
    
    # --- ĐỘNG CƠ 2: BETA AR ---
    p_t = p_raw[1:]
    p_tm1 = p_raw[:-1]
    cov_matrix = np.cov(p_t, p_tm1)
    phi_beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-10)
    phi_beta = np.clip(phi_beta, -0.95, 0.95)
    mu_beta = p_raw.mean()
    
    resid_beta = p_t - (mu_beta * (1 - phi_beta) + phi_beta * p_tm1)
    sigma_beta = max(resid_beta.std(), 0.05)
    
    sim_p_beta = np.zeros((days_to_sim, num_sims))
    sim_p_beta[0, :] = last_p
    for t in range(1, days_to_sim):
        mean_t = mu_beta * (1 - phi_beta) + phi_beta * sim_p_beta[t-1, :]
        mean_t = np.clip(mean_t, 0.001, 0.999)
        kappa = mean_t * (1 - mean_t) / (sigma_beta ** 2) - 1
        kappa = np.maximum(kappa, 0.5)
        sim_p_beta[t, :] = np.random.beta(mean_t * kappa, (1 - mean_t) * kappa)
        
    # --- ENSEMBLE POOLING ---
    pooled_sim_p = np.hstack((sim_p_emp, sim_p_beta)) * 100.0
    total_sims = num_sims * 2
    
    past_4_days = raw_ratio_series.values[-4:]
    past_4_matrix = np.tile(past_4_days, (total_sims, 1)).T
    full_sim_p = np.vstack((past_4_matrix, pooled_sim_p))
    
    sim_ma5 = np.zeros_like(pooled_sim_p)
    for t in range(days_to_sim):
        sim_ma5[t, :] = np.mean(full_sim_p[t:t+5, :], axis=0)
        
    p5  = np.percentile(sim_ma5,  5, axis=1)
    p25 = np.percentile(sim_ma5, 25, axis=1)
    p50 = np.percentile(sim_ma5, 50, axis=1)
    p75 = np.percentile(sim_ma5, 75, axis=1)
    p95 = np.percentile(sim_ma5, 95, axis=1)
    
    return p5, p25, p50, p75, p95, phi_beta, mu_beta, resid_emp, resid_beta

# ==========================================
# 3. XỬ LÝ THỜI GIAN & GIAO DIỆN TIÊU ĐỀ
# ==========================================
st.title("🧬 Hybrid MC Bidirectional Breadth Model")

df_cache = load_cached_prices()
df_index_cache = load_cached_vnindex()

# HIỂN THỊ NGÀY CẬP NHẬT CUỐI CÙNG NGAY DƯỚI TIÊU ĐỀ
if not df_cache.empty:
    last_date_in_cache = df_cache.index.max().date()
    st.caption(f"📅 **Dữ liệu chốt phiên gần nhất:** {last_date_in_cache.strftime('%d/%m/%Y')}")
else:
    st.caption("📅 **Dữ liệu chưa được tải. Vui lòng cập nhật ở thanh công cụ.**")

now = datetime.now()
if now.hour < 15:
    actual_today = now - timedelta(days=1)
else:
    actual_today = now

today_str = actual_today.strftime('%Y-%m-%d')
today_date = actual_today.date()

# ==========================================
# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
with st.expander("🗄️ Quản lý Dữ liệu (Local Cache)", expanded=False):
    if df_cache.empty or df_index_cache.empty:
        st.warning("Chưa có đủ dữ liệu cục bộ.")
        if st.button("📥 Tải dữ liệu ban đầu"):
            start_date_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            new_data = fetch_prices_kbs(DEFAULT_SYMBOLS, start_date_str, today_str)
            if not new_data.empty:
                new_data.to_csv(CSV_FILE)
            st.info("Đang tải dữ liệu VN-Index...")
            new_index = fetch_vnindex_data(start_date_str, today_str)
            if not new_index.empty:
                new_index.to_csv(VNINDEX_FILE)
            
            # --- FIX LỖI CACHE Ở ĐÂY ---
            load_cached_prices.clear()
            load_cached_vnindex.clear()
            # ---------------------------
            
            st.rerun()
    else:
        last_date_in_cache = df_cache.index.max().date()
        st.success(f"Dữ liệu cục bộ OK — Cập nhật lần cuối: **{last_date_in_cache}**")
        
        # --- FIX GIAO DIỆN: Luôn hiện nút, không bao giờ ẩn đi ---
        if last_date_in_cache < today_date:
            btn_label = "🔄 Cập nhật dữ liệu mới"
            btn_style = "primary"
            start_fetch = last_date_in_cache.strftime('%Y-%m-%d')
        else:
            btn_label = "⚠️ Ép tải lại (Force Update)"
            btn_style = "secondary"
            # Lùi lại 1 ngày để API quét đè lại dữ liệu phiên hôm nay
            start_fetch = (today_date - timedelta(days=1)).strftime('%Y-%m-%d')

        if st.button(btn_label, type=btn_style):
            new_data = fetch_prices_kbs(DEFAULT_SYMBOLS, start_fetch, today_str)
            if not new_data.empty:
                df_combined = pd.concat([df_cache, new_data])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                df_combined.to_csv(CSV_FILE)
            
            st.info("Đang tải dữ liệu VN-Index mới...")
            new_index = fetch_vnindex_data(start_fetch, today_str)
            if not new_index.empty:
                index_combined = pd.concat([df_index_cache, new_index])
                index_combined = index_combined[~index_combined.index.duplicated(keep='last')]
                index_combined.sort_index(inplace=True)
                index_combined.to_csv(VNINDEX_FILE)
            
            load_cached_prices.clear()
            load_cached_vnindex.clear()
            st.rerun()

with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    user_X = st.number_input("Ngưỡng Upside X (%)", value=2.0, step=0.5)
    user_Y = st.number_input("Ngưỡng Downside Y (%)", value=-2.0, step=0.5)
    
    lookback_days = st.slider("Khung lịch sử (ngày)", min_value=30, max_value=250, value=90)
    sim_days = st.slider("Số phiên giả lập", min_value=5, max_value=40, value=10)

    st.divider()
    st.header("⏳ Cỗ máy thời gian (Backtest)")
    run_mode = st.radio("Chế độ chạy:", ["Live (Hiện tại)", "Backtest (Quá khứ)"], horizontal=True)
    
    backtest_date = None
    if run_mode == "Backtest (Quá khứ)" and not df_cache.empty:
        max_bt_date = df_cache.index.max().date() - timedelta(days=sim_days)
        min_bt_date = df_cache.index.min().date() + timedelta(days=lookback_days + 10)
        backtest_date = st.date_input("Chọn ngày muốn quay về:", value=max_bt_date, min_value=min_bt_date, max_value=max_bt_date)

    # NÂNG CẤP: HIỂN THỊ SONG SONG 2 ĐỘNG CƠ
    st.divider()
    st.subheader("📈 Lực Cầu (Upside Momentum)")
    side_up_phi = st.empty()
    side_up_mu = st.empty()
    
    st.divider()
    st.subheader("🩸 Lực Cung (Downside Momentum)")
    side_dn_phi = st.empty()
    side_dn_mu = st.empty()

    st.divider()
    st.subheader("🧠 Tích hợp AI")
    gemini_key = st.text_input("Gemini API Key (Bảo mật)", type="password", value="AIzaSyCGxCEprEKDB-f9n6CmXzdFTlJ6u4invqA")

st.divider()

if not df_cache.empty and not df_index_cache.empty:
    df_returns = df_cache.pct_change() * 100
    df_returns = df_returns.dropna(how='all')
    
    valid_trading_days = df_returns.abs().sum(axis=1) > 0.001
    df_returns = df_returns.loc[valid_trading_days]

    if run_mode == "Backtest (Quá khứ)" and backtest_date is not None:
        target_date_pd = pd.to_datetime(backtest_date)
        train_returns = df_returns.loc[:target_date_pd].tail(lookback_days + 10)
        future_returns_raw = df_returns.loc[:].copy()
    else:
        train_returns = df_returns.tail(lookback_days + 10)
        future_returns_raw = pd.DataFrame()

    total_valid = train_returns.notna().sum(axis=1)
    
    # --- TÍNH UPSIDE ---
    upside_counts = (train_returns > user_X).sum(axis=1)
    raw_upside = (upside_counts / total_valid) * 100
    ma5_upside = raw_upside.rolling(window=5).mean().dropna().tail(lookback_days)
    raw_upside = raw_upside.loc[ma5_upside.index]
    
    # --- TÍNH DOWNSIDE ---
    downside_counts = (train_returns < user_Y).sum(axis=1)
    raw_downside = (downside_counts / total_valid) * 100
    ma5_downside = raw_downside.rolling(window=5).mean().dropna().tail(lookback_days)
    raw_downside = raw_downside.loc[ma5_downside.index]

    # ==========================================
    # CHẠY ĐỘNG CƠ KÉP (DUAL ENGINE)
    # ==========================================
    # Động cơ 1: Cầu
    p5_up, p25_up, p50_up, p75_up, p95_up, phi_up, mu_up, resid_emp_up, resid_beta_up = run_hybrid_ensemble_mc(raw_upside, sim_days, 3000)
    regime_up = ("📈 Momentum (Đà Mua)" if phi_up > 0.1 else "🔄 Mean-reversion (Đảo chiều Mua)" if phi_up < -0.1 else "🎲 Random Walk (Nhiễu Mua)")
    
    # Động cơ 2: Cung
    p5_dn, p25_dn, p50_dn, p75_dn, p95_dn, phi_dn, mu_dn, resid_emp_dn, resid_beta_dn = run_hybrid_ensemble_mc(raw_downside, sim_days, 3000)
    regime_dn = ("🩸 Momentum (Đà Bán)" if phi_dn > 0.1 else "🔄 Mean-reversion (Đảo chiều Bán)" if phi_dn < -0.1 else "🎲 Random Walk (Nhiễu Bán)")

    # Đẩy số liệu ra Sidebar
    side_up_phi.metric("Core Momentum Cầu (φ)", f"{phi_up:.3f}", regime_up)
    side_up_mu.metric("Long-run Mean Cầu (μ)", f"{mu_up*100:.1f}%")
    
    side_dn_phi.metric("Core Momentum Cung (φ)", f"{phi_dn:.3f}", regime_dn)
    side_dn_mu.metric("Long-run Mean Cung (μ)", f"{mu_dn*100:.1f}%")

    # ==========================================
    # BIỂU ĐỒ 1: LỊCH SỬ CUNG - CẦU & VN-INDEX
    # ==========================================
    header_text = "1. Lịch sử Cung - Cầu & VN-Index "
    if run_mode == "Backtest (Quá khứ)":
         header_text += f"(Góc nhìn từ ngày {backtest_date.strftime('%d/%m/%Y')})"
    st.subheader(header_text)

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(go.Bar(x=raw_upside.index, y=raw_upside.values, name='Raw Upside (Cầu)', marker_color='rgba(37, 99, 235, 0.25)'), secondary_y=False)
    fig1.add_trace(go.Bar(x=raw_downside.index, y=raw_downside.values, name='Raw Downside (Cung)', marker_color='rgba(239, 68, 68, 0.25)'), secondary_y=False)
    
    fig1.add_trace(go.Scatter(x=ma5_upside.index, y=ma5_upside.values, mode='lines', name='MA5 Upside', line=dict(color='#2563eb', width=2.5)), secondary_y=False)
    fig1.add_trace(go.Scatter(x=ma5_downside.index, y=ma5_downside.values, mode='lines', name='MA5 Downside', line=dict(color='#ef4444', width=2.5)), secondary_y=False)
    
    vnindex_plot_data = df_index_cache.loc[df_index_cache.index.isin(ma5_upside.index)]
    if not vnindex_plot_data.empty:
        fig1.add_trace(go.Scatter(x=vnindex_plot_data.index, y=vnindex_plot_data['VNINDEX'], mode='lines', name='VN-Index', line=dict(color='rgba(34, 197, 94, 0.9)', width=2, dash='dashdot')), secondary_y=True)

    # Đường tham chiếu
    fig1.add_hline(y=mu_up * 100, line=dict(color='#2563eb', width=1, dash='dot'), secondary_y=False)
    fig1.add_hline(y=mu_dn * 100, line=dict(color='#ef4444', width=1, dash='dot'), secondary_y=False)
    
    fig1.update_layout(barmode='group', hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig1.update_yaxes(title_text="Breadth Ratio (%)", secondary_y=False)
    fig1.update_yaxes(title_text="VN-Index", secondary_y=True, showgrid=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ==========================================
    # ==========================================
    # BIỂU ĐỒ 2: DỰ PHÓNG CUNG - CẦU (TÁCH TAB)
    # ==========================================
    st.subheader(f"2. Dự phóng Monte Carlo — 6.000 kịch bản × {sim_days} phiên")

    # Tạo 2 Tab riêng biệt trên giao diện
    tab_up, tab_dn = st.tabs(["📈 Dự phóng CẦU (Upside)", "🩸 Dự phóng CUNG (Downside)"])

    last_date = raw_upside.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=sim_days).date.tolist()
    n_tail = 10
    past_dates = list(raw_upside.index[-n_tail:])

    # ----------------------------------
    # TAB 1: VẼ DỰ PHÓNG UPSIDE (MÀU XANH/CAM)
    # ----------------------------------
    with tab_up:
        past_values_raw_up = list(raw_upside.values[-n_tail:])
        past_values_ma5_up = list(ma5_upside.values[-n_tail:])

        full_dates = past_dates + future_dates
        full_p5_up  = past_values_ma5_up + list(p5_up)
        full_p25_up = past_values_ma5_up + list(p25_up)
        full_p50_up = past_values_ma5_up + list(p50_up)
        full_p75_up = past_values_ma5_up + list(p75_up)
        full_p95_up = past_values_ma5_up + list(p95_up)

        fig_up = go.Figure()
        fig_up.add_trace(go.Scatter(x=full_dates, y=full_p5_up, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_up.add_trace(go.Scatter(x=full_dates, y=full_p95_up, mode='lines', fill='tonexty', fillcolor='rgba(250,160,60,0.15)', line=dict(width=0), name='Dải 90% (Fat-Tail)'))
        fig_up.add_trace(go.Scatter(x=full_dates, y=full_p25_up, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_up.add_trace(go.Scatter(x=full_dates, y=full_p75_up, mode='lines', fill='tonexty', fillcolor='rgba(250,160,60,0.35)', line=dict(width=0), name='Dải 50% (Core)'))
        fig_up.add_hline(y=mu_up * 100, line=dict(color='orange', width=1.5, dash='dot'), annotation_text=f"Long-run μ = {mu_up*100:.1f}%")
        
        fig_up.add_trace(go.Scatter(x=past_dates, y=past_values_raw_up, mode='lines+markers', name='Lịch sử (Raw Upside)', line=dict(color='rgba(37, 99, 235, 0.4)', width=2)))
        fig_up.add_trace(go.Scatter(x=past_dates, y=past_values_ma5_up, mode='lines', name='MA5 Upside', line=dict(color='#2563eb', width=3)))
        fig_up.add_trace(go.Scatter(x=full_dates, y=full_p50_up, mode='lines', name='Ensemble Median', line=dict(color='#dc2626', width=3, dash='dash')))

        # Backtest cho Upside
        if run_mode == "Backtest (Quá khứ)":
            try:
                target_idx = future_returns_raw.index.get_loc(target_date_pd)
                actual_future_data = future_returns_raw.iloc[target_idx - 4 : target_idx + sim_days + 1]
                actual_upside_counts = (actual_future_data > user_X).sum(axis=1)
                actual_total_valid = actual_future_data.notna().sum(axis=1)
                actual_raw_up = (actual_upside_counts / actual_total_valid) * 100
                actual_ma5_up = actual_raw_up.rolling(window=5).mean().dropna()
                
                fig_up.add_trace(go.Scatter(x=actual_ma5_up.index, y=actual_ma5_up.values, mode='lines+markers', name='🚀 THỰC TẾ CẦU DIỄN RA', line=dict(color='black', width=4)))
                fig_up.add_vline(x=target_date_pd.timestamp() * 1000, line=dict(color='black', dash='dash'), annotation_text="Ngày Backtest")
            except KeyError:
                pass

        fig_up.update_layout(hovermode="x unified", yaxis_title="Upside Ratio (%)", yaxis=dict(range=[0, min(100, max(full_p95_up)*1.2)]), margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_up, use_container_width=True)

    # ----------------------------------
    # TAB 2: VẼ DỰ PHÓNG DOWNSIDE (MÀU ĐỎ/TÍM)
    # ----------------------------------
    with tab_dn:
        past_values_raw_dn = list(raw_downside.values[-n_tail:])
        past_values_ma5_dn = list(ma5_downside.values[-n_tail:])

        full_p5_dn  = past_values_ma5_dn + list(p5_dn)
        full_p25_dn = past_values_ma5_dn + list(p25_dn)
        full_p50_dn = past_values_ma5_dn + list(p50_dn)
        full_p75_dn = past_values_ma5_dn + list(p75_dn)
        full_p95_dn = past_values_ma5_dn + list(p95_dn)

        fig_dn = go.Figure()
        fig_dn.add_trace(go.Scatter(x=full_dates, y=full_p5_dn, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_dn.add_trace(go.Scatter(x=full_dates, y=full_p95_dn, mode='lines', fill='tonexty', fillcolor='rgba(239, 68, 68, 0.15)', line=dict(width=0), name='Dải 90% Cung (Fat-Tail)'))
        fig_dn.add_trace(go.Scatter(x=full_dates, y=full_p25_dn, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_dn.add_trace(go.Scatter(x=full_dates, y=full_p75_dn, mode='lines', fill='tonexty', fillcolor='rgba(239, 68, 68, 0.35)', line=dict(width=0), name='Dải 50% Cung (Core)'))
        fig_dn.add_hline(y=mu_dn * 100, line=dict(color='#ef4444', width=1.5, dash='dot'), annotation_text=f"Long-run μ = {mu_dn*100:.1f}%")
        
        fig_dn.add_trace(go.Scatter(x=past_dates, y=past_values_raw_dn, mode='lines+markers', name='Lịch sử (Raw Downside)', line=dict(color='rgba(239, 68, 68, 0.4)', width=2)))
        fig_dn.add_trace(go.Scatter(x=past_dates, y=past_values_ma5_dn, mode='lines', name='MA5 Downside', line=dict(color='#ef4444', width=3)))
        fig_dn.add_trace(go.Scatter(x=full_dates, y=full_p50_dn, mode='lines', name='Ensemble Median', line=dict(color='#991b1b', width=3, dash='dash')))

        # Backtest cho Downside
        if run_mode == "Backtest (Quá khứ)":
            try:
                target_idx = future_returns_raw.index.get_loc(target_date_pd)
                actual_future_data = future_returns_raw.iloc[target_idx - 4 : target_idx + sim_days + 1]
                # CHÚ Ý: Đếm số cổ phiếu < user_Y (Downside)
                actual_downside_counts = (actual_future_data < user_Y).sum(axis=1) 
                actual_total_valid = actual_future_data.notna().sum(axis=1)
                actual_raw_dn = (actual_downside_counts / actual_total_valid) * 100
                actual_ma5_dn = actual_raw_dn.rolling(window=5).mean().dropna()
                
                fig_dn.add_trace(go.Scatter(x=actual_ma5_dn.index, y=actual_ma5_dn.values, mode='lines+markers', name='🩸 THỰC TẾ CUNG DIỄN RA', line=dict(color='black', width=4)))
                fig_dn.add_vline(x=target_date_pd.timestamp() * 1000, line=dict(color='black', dash='dash'), annotation_text="Ngày Backtest")
            except KeyError:
                pass

        fig_dn.update_layout(hovermode="x unified", yaxis_title="Downside Ratio (%)", yaxis=dict(range=[0, min(100, max(full_p95_dn)*1.2)]), margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_dn, use_container_width=True)

    # ==========================================
    # DIAGNOSTICS (CHIA TAB CHO CẢ 2 ĐỘNG CƠ)
    # ==========================================
    with st.expander("🔬 Model Diagnostics — Beta fit & Residuals", expanded=False):
        tab1, tab2 = st.tabs(["📈 Chuẩn đoán Upside", "📉 Chuẩn đoán Downside"])
        
        # --- TAB 1: UPSIDE ---
        with tab1:
            col1, col2 = st.columns(2)
            p_hist_up = np.clip(raw_upside.values / 100.0, 0.001, 0.999)
            a_fit_up, b_fit_up, _, _ = stats.beta.fit(p_hist_up, floc=0, fscale=1)

            with col1:
                fig_hist_up = go.Figure()
                fig_hist_up.add_trace(go.Histogram(x=raw_upside.values, nbinsx=30, histnorm='probability density', marker_color='#3b82f6', opacity=0.65, name='Thực tế'))
                x_range = np.linspace(0.5, 99.5, 300)
                beta_pdf = stats.beta.pdf(x_range / 100, a_fit_up, b_fit_up) / 100
                fig_hist_up.add_trace(go.Scatter(x=x_range, y=beta_pdf, mode='lines', line=dict(color='orange', width=2.5), name=f'Beta fit'))
                fig_hist_up.update_layout(xaxis_title="Upside Ratio (%)", margin=dict(l=10, r=10, t=20, b=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_hist_up, use_container_width=True)

            with col2:
                fig_resid_up = go.Figure()
                fig_resid_up.add_trace(go.Histogram(x=resid_beta_up * 100, nbinsx=30, histnorm='probability density', marker_color='#8b5cf6', opacity=0.65, name='Sốc Thực tế'))
                xr = np.linspace(resid_beta_up.min()*100, resid_beta_up.max()*100, 300)
                norm_pdf = stats.norm.pdf(xr, resid_beta_up.mean()*100, resid_beta_up.std()*100)
                fig_resid_up.add_trace(go.Scatter(x=xr, y=norm_pdf, mode='lines', line=dict(color='red', width=2, dash='dash'), name='Normal'))
                fig_resid_up.update_layout(xaxis_title="Residual (%)", margin=dict(l=10, r=10, t=20, b=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_resid_up, use_container_width=True)
                
            ks_stat_up, ks_pval_up = stats.kstest(p_hist_up, 'beta', args=(a_fit_up, b_fit_up))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("KS Statistic", f"{ks_stat_up:.3f}")
            c2.metric("KS p-value", f"{ks_pval_up:.3f}")

        # --- TAB 2: DOWNSIDE ---
        with tab2:
            col3, col4 = st.columns(2)
            p_hist_dn = np.clip(raw_downside.values / 100.0, 0.001, 0.999)
            a_fit_dn, b_fit_dn, _, _ = stats.beta.fit(p_hist_dn, floc=0, fscale=1)

            with col3:
                fig_hist_dn = go.Figure()
                fig_hist_dn.add_trace(go.Histogram(x=raw_downside.values, nbinsx=30, histnorm='probability density', marker_color='#ef4444', opacity=0.65, name='Thực tế'))
                beta_pdf_dn = stats.beta.pdf(x_range / 100, a_fit_dn, b_fit_dn) / 100
                fig_hist_dn.add_trace(go.Scatter(x=x_range, y=beta_pdf_dn, mode='lines', line=dict(color='orange', width=2.5), name=f'Beta fit'))
                fig_hist_dn.update_layout(xaxis_title="Downside Ratio (%)", margin=dict(l=10, r=10, t=20, b=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_hist_dn, use_container_width=True)

            with col4:
                fig_resid_dn = go.Figure()
                fig_resid_dn.add_trace(go.Histogram(x=resid_beta_dn * 100, nbinsx=30, histnorm='probability density', marker_color='#f97316', opacity=0.65, name='Sốc Thực tế'))
                xr_dn = np.linspace(resid_beta_dn.min()*100, resid_beta_dn.max()*100, 300)
                norm_pdf_dn = stats.norm.pdf(xr_dn, resid_beta_dn.mean()*100, resid_beta_dn.std()*100)
                fig_resid_dn.add_trace(go.Scatter(x=xr_dn, y=norm_pdf_dn, mode='lines', line=dict(color='red', width=2, dash='dash'), name='Normal'))
                fig_resid_dn.update_layout(xaxis_title="Residual (%)", margin=dict(l=10, r=10, t=20, b=20), legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_resid_dn, use_container_width=True)
                
            ks_stat_dn, ks_pval_dn = stats.kstest(p_hist_dn, 'beta', args=(a_fit_dn, b_fit_dn))
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("KS Statistic", f"{ks_stat_dn:.3f}")
            c6.metric("KS p-value", f"{ks_pval_dn:.3f}")

    # ==========================================
    # KẾT LUẬN & TRỢ LÝ AI (NÂNG CẤP PROMPT)
    # ==========================================
    # ==========================================
    # KẾT LUẬN & TRỢ LÝ AI (CHI TIẾT 2 CHIỀU)
    # ==========================================
    
    # Logic Hồi quy cho Lực Cầu
    current_vs_mu_up = "thấp hơn" if raw_upside.values[-1] < mu_up*100 else "cao hơn"
    mean_reversion_note_up = f"→ Khả năng **hồi phục**" if raw_upside.values[-1] < mu_up * 100 else f"→ Khả năng **điều chỉnh**"

    # Logic Hồi quy cho Lực Cung (Ngược lại: Cung tăng là xấu, Cung giảm là tốt)
    current_vs_mu_dn = "thấp hơn" if raw_downside.values[-1] < mu_dn*100 else "cao hơn"
    mean_reversion_note_dn = f"→ Khả năng **gia tăng áp lực bán**" if raw_downside.values[-1] < mu_dn * 100 else f"→ Khả năng **hạ nhiệt bán tháo**"

    # Khung hiển thị Lực Cầu (Màu Xanh)
    st.info(
        f"**📈 DỰ PHÓNG LỰC CẦU / UPSIDE (T+{sim_days-1}):**\n\n"
        f"Ensemble Median = **{p50_up[-1]:.1f}%** &nbsp;|&nbsp; "
        f"Dải Core 50%: **{p25_up[-1]:.1f}% – {p75_up[-1]:.1f}%** &nbsp;|&nbsp; "
        f"Dải Rủi ro 90%: **{p5_up[-1]:.1f}% – {p95_up[-1]:.1f}%**\n\n"
        f"Giá trị tính đến ngày chốt **{raw_upside.values[-1]:.1f}%** đang {current_vs_mu_up} Long-run mean {mu_up*100:.1f}%. {mean_reversion_note_up} về trung bình dài hạn."
    )

    # Khung hiển thị Lực Cung (Màu Đỏ)
    st.error(
        f"**🩸 DỰ PHÓNG LỰC CUNG / DOWNSIDE (T+{sim_days-1}):**\n\n"
        f"Ensemble Median = **{p50_dn[-1]:.1f}%** &nbsp;|&nbsp; "
        f"Dải Core 50%: **{p25_dn[-1]:.1f}% – {p75_dn[-1]:.1f}%** &nbsp;|&nbsp; "
        f"Dải Rủi ro 90%: **{p5_dn[-1]:.1f}% – {p95_dn[-1]:.1f}%**\n\n"
        f"Giá trị tính đến ngày chốt **{raw_downside.values[-1]:.1f}%** đang {current_vs_mu_dn} Long-run mean {mu_dn*100:.1f}%. {mean_reversion_note_dn} về trung bình dài hạn."
    )

    st.divider()
    st.subheader("✨ Trợ lý AI Quant Phân tích Đa chiều")
    
    # (Đoạn code if st.button("🤖 Phân tích Rủi ro 2 chiều... giữ nguyên như cũ)

    if st.button("🤖 Phân tích Rủi ro 2 chiều (Gemini AI)", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("⚠️ Bạn chưa nhập Gemini API Key ở thanh menu bên trái.")
        else:
            with st.spinner("AI đang quét ma trận 12.000 kịch bản Cung - Cầu..."):
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')

                    prompt = f"""
                    Bạn là Giám đốc Quản trị Rủi ro (CRO) tại một quỹ lượng hóa. 
                    Mô hình của chúng ta vừa chạy 12.000 kịch bản Monte Carlo (Hybrid Beta-AR & Bootstrap) cho cả 2 chiều Cung (Downside) và Cầu (Upside). Hãy đọc dữ liệu và lên phương án tác chiến.
                    
                    **DỮ LIỆU ĐẦU VÀO (Thời điểm hiện tại):**
                    - [LỰC CẦU - Upside]: Hiện tại: {raw_upside.values[-1]:.2f}%. Trung bình dài hạn (Mu): {mu_up*100:.2f}%. Quán tính (Phi): {phi_up:.3f} ({regime_up}).
                    - [LỰC CUNG - Downside]: Hiện tại: {raw_downside.values[-1]:.2f}%. Trung bình dài hạn (Mu): {mu_dn*100:.2f}%. Quán tính (Phi): {phi_dn:.3f} ({regime_dn}).
                    
                    **DỰ PHÓNG RỦI RO ĐUÔI (Tail Risk T+{sim_days-1}):**
                    - Kịch bản Bùng nổ (P95 Upside): Dòng tiền mua lan tỏa cực đại lên đến {p95_up[-1]:.2f}%.
                    - Kịch bản Thảm họa (P95 Downside): Lực bán tháo hoảng loạn có thể vọt lên {p95_dn[-1]:.2f}%.
                    *(Lưu ý: Đối với Downside, tỷ lệ phần trăm càng cao nghĩa là rủi ro càng lớn).*

                    **NHIỆM VỤ PHÂN TÍCH (Lập luận sắt đá, lạnh lùng):**
                    1. Cuộc chiến Cung - Cầu: Nhìn vào sự chênh lệch giữa Upside hiện tại so với Mu, và Downside hiện tại so với Mu. Dòng tiền đang ở trạng thái Zombification (Nén), Tích lũy, hay Phân phối?
                    2. Đụng độ Quán tính (Momentum Clash): Phân tích hệ số Phi của 2 bên. Bên nào (Cung hay Cầu) đang giữ được gia tốc thực sự? Hay cả 2 đang bị nhiễu (Random Walk)?
                    3. Stress-Test: Dựa vào P95 Downside, mức độ lan tỏa hoảng loạn (tỷ lệ % số mã trên sàn bị xả rát) tiềm ẩn trong những phiên tới là bao nhiêu? (Tuyệt đối không nhầm lẫn con số này với mức lỗ Max Drawdown của danh mục).
                    4. Lệnh Tác Chiến: Đưa ra chiến lược cụ thể (Tỷ trọng giải ngân, Ưu tiên phòng thủ hay tấn công, Mua đuổi hay rình bắt đáy).
                    
                    Viết chuyên nghiệp, chia 4 gạch đầu dòng rõ ràng.
                    """

                    response = model.generate_content(prompt)
                    st.success("Hoàn thành phân tích!")
                    with st.container(border=True):
                        st.markdown(response.text)

                except Exception as e:
                    st.error(f"Lỗi kết nối API: {e}. Vui lòng kiểm tra lại API Key.")
