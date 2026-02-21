# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 11:32:27 2026

@author: cylin
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings

# 忽略不必要的警告 (如 FutureWarning)，保持介面乾淨
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 設定與樣式
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="股市崩盤模式比對系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 讓介面看起來更專業
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30333d;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stAlert {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. 資料處理與演算法函數
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_data(symbol, start_date, end_date):
    """
    從 yfinance 抓取每日 OHLCV 資料。
    關鍵修正：加入 threads=False 以防止在 Streamlit 環境下出現 'missing ScriptRunContext' 警告。
    """
    try:
        # 下載資料
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=False)
        
        if df.empty:
            return None
        
        # 處理 MultiIndex 欄位 (yfinance 新版常見問題，移除多層索引)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        # 統一欄位名稱為小寫
        df.columns = [c.lower() for c in df.columns]
        # 重新命名為標準格式
        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        st.error(f"資料抓取錯誤: {e}")
        return None

def calculate_indicators(df):
    """
    計算移動平均線 (MA) 與 最大回撤 (Max Drawdown)。
    """
    df = df.copy()
    
    # 計算移動平均線 (30, 50, 200日)
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 計算滾動最高價 (Rolling Peak)
    df['RollingMax'] = df['Close'].expanding().max()
    
    # 計算回撤 (Drawdown): (當前價格 - 滾動最高價) / 滾動最高價
    df['Drawdown'] = (df['Close'] / df['RollingMax']) - 1
    
    return df

def find_top_crashes(df, n=10, min_separation_days=180):
    """
    找出歷史上跌幅最深的 N 次崩盤。
    
    邏輯:
    1. 找出 Drawdown 曲線的局部極小值 (Local Minima)。
    2. 依照跌幅深度排序。
    3. 過濾掉時間太相近的事件 (避免同一次崩盤被重複計算)。
    4. 回傳該次崩盤前的「起跌點 (Pivot High)」日期。
    """
    dd_series = df['Drawdown']
    
    # 找出比前後都低的點 (波谷)
    is_local_min = (dd_series < dd_series.shift(1)) & (dd_series < dd_series.shift(-1))
    troughs = dd_series[is_local_min]
    
    # 依照跌幅排序 (越小越深，所以由小到大排)
    sorted_troughs = troughs.sort_values(ascending=True)
    
    unique_crashes = []
    
    # 分離邏輯：避免選到同一次崩盤的連續低點
    for date, depth in sorted_troughs.items():
        is_distinct = True
        for existing_crash in unique_crashes:
            # 如果這一天跟已經選出的崩盤日期太近，就跳過
            if abs((date - existing_crash['trough_date']).days) < min_separation_days:
                is_distinct = False
                break
        
        if is_distinct:
            # 找出這次崩盤前的最高點 (起跌點)
            # 也就是 Drawdown 接近 0 的最後一天
            subset = df.loc[:date]
            # 確保有資料，避免空值錯誤
            if not subset.empty and (subset['Drawdown'] == 0).any():
                peak_date = subset[subset['Drawdown'] == 0].index[-1]
                
                unique_crashes.append({
                    'peak_date': peak_date,   # 起跌點
                    'trough_date': date,      # 最低點
                    'depth': depth            # 跌幅
                })
            
        if len(unique_crashes) >= n:
            break
            
    return unique_crashes

def get_normalized_series(series):
    """
    將價格序列標準化到 [0, 1] 之間。
    這是為了比較「形狀」而非絕對價格。
    """
    scaler = MinMaxScaler()
    values = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values).flatten()
    return scaled

def calculate_similarity(current_seq, hist_seq):
    """
    計算兩個價格序列的相似度。
    
    步驟:
    1. 線性插值 (Interpolation)：將歷史序列長度調整為與當前序列一致。
    2. 標準化 (Normalization)：將兩者都縮放到 0-1。
    3. 歐幾里得距離 (Euclidean Distance)：計算幾何距離。
    4. 轉換為相似度分數 (0-100%)。
    """
    target_len = len(current_seq)
    
    # 1. 插值/重採樣
    x_hist = np.linspace(0, 1, len(hist_seq))
    x_target = np.linspace(0, 1, target_len)
    hist_interpolated = np.interp(x_target, x_hist, hist_seq)
    
    # 2. 標準化
    norm_curr = get_normalized_series(pd.Series(current_seq))
    norm_hist = get_normalized_series(pd.Series(hist_interpolated))
    
    # 3. 計算距離
    dist = np.linalg.norm(norm_curr - norm_hist)
    
    # 4. 轉換為分數
    # 兩向量長度為 N 且值在 [0,1]，最大可能距離為 sqrt(N)
    max_dist = np.sqrt(target_len)
    similarity = 100 * (1 - (dist / max_dist))
    
    return max(0, similarity)

# -----------------------------------------------------------------------------
# 2. 視覺化函數
# -----------------------------------------------------------------------------

def create_comprehensive_chart(df, title):
    """
    建立包含三個子圖的圖表：K線圖、成交量、回撤幅度。
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{title} - 價格走勢", "成交量", "最大回撤幅度")
    )
    
    # 第一行: K線圖與均線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    colors = {'MA30': 'cyan', 'MA50': 'orange', 'MA200': 'purple'}
    for ma, color in colors.items():
        if ma in df.columns:
            # 過濾掉 NaN 值以避免繪圖中斷
            ma_data = df[ma].dropna()
            if not ma_data.empty:
                fig.add_trace(go.Scatter(
                    x=ma_data.index, y=ma_data, name=ma, line=dict(color=color, width=1)
                ), row=1, col=1)

    # 第二行: 成交量
    colors_vol = ['red' if c < o else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol
    ), row=2, col=1)

    # 第三行: 回撤
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Drawdown'], name='Drawdown',
        fill='tozeroy', line=dict(color='red', width=1)
    ), row=3, col=1)

    # 版面設定
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# -----------------------------------------------------------------------------
# 3. 主程式邏輯
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("📉 股市崩盤預警系統")
    st.sidebar.markdown("---")
    
    # --- 側邊欄輸入 ---
    # 提供常見代碼建議
    symbol = st.sidebar.text_input("股票代碼 (Symbol)", value="^NDX", help="預設為 Nasdaq 100 (^NDX)。台股請加 .TW (如 2330.TW)。")
    
    # 日期處理
    today = datetime.now().date()
    # 預設分析日期 (如果今天比預設還早，就用今天)
    default_target = datetime(2026, 2, 13).date()
    input_date_val = default_target if default_target <= today else today
    
    input_date = st.sidebar.date_input(
        "當前分析基準日 (Analysis Date)", 
        value=input_date_val,
        max_value=today
    )
    
    lookback_days = st.sidebar.slider("回溯比對天數 (Lookback Period)", 30, 120, 60)
    
    st.sidebar.subheader("歷史資料設定")
    start_year = st.sidebar.number_input("歷史資料起始年份", 1990, 2025, 2007)
    
    # --- 資料抓取 ---
    # 歷史資料範圍：從使用者設定的年份開始，直到分析日期的隔天
    hist_start = f"{start_year}-01-01"
    hist_end = (pd.to_datetime(input_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    with st.spinner(f"正在抓取 {symbol} 的資料..."):
        df = fetch_data(symbol, hist_start, hist_end)
    
    if df is None:
        st.error("找不到資料，請檢查股票代碼是否正確。")
        st.info("提示：美股如 ^GSPC, ^NDX, AAPL；台股如 2330.TW")
        return

    # 計算指標
    df = calculate_indicators(df)

    # --- 提取當前模式 ---
    # 篩選出截至「分析日期」為止的資料
    current_mask = (df.index.date <= input_date)
    df_current_full = df.loc[current_mask]
    
    if len(df_current_full) < lookback_days:
        st.error(f"資料不足，無法進行回溯分析。可用天數: {len(df_current_full)} 天，需要: {lookback_days} 天。")
        return

    # 提取最後 N 天的收盤價作為「當前模式」
    current_pattern_df = df_current_full.iloc[-lookback_days:]
    current_sequence = current_pattern_df['Close'].values

    # --- 尋找歷史崩盤 ---
    # 搜尋範圍排除掉最近這段時間 (避免自己跟自己比對)
    cutoff_date = pd.to_datetime(input_date) - timedelta(days=lookback_days*2)
    df_history_search = df.loc[df.index < cutoff_date]
    
    # 找出前 10 大崩盤
    top_crashes = find_top_crashes(df_history_search, n=10)
    
    if not top_crashes:
        st.warning("在指定的歷史範圍內未發現顯著崩盤。請嘗試將起始年份調早。")
        return

    # --- 計算相似度 ---
    results = []
    
    for crash in top_crashes:
        peak_date = crash['peak_date']
        
        # 定義歷史樣本視窗：以起跌點 (Peak) 為中心，前後各取 N/2 天
        window_half = int(lookback_days / 2)
        
        try:
            peak_loc = df.index.get_loc(peak_date)
            # 確保索引不越界
            start_loc = max(0, peak_loc - window_half)
            end_loc = min(len(df), peak_loc + window_half)
            
            hist_slice = df.iloc[start_loc:end_loc]
            hist_sequence = hist_slice['Close'].values
            
            # 計算相似度
            score = calculate_similarity(current_sequence, hist_sequence)
            
            results.append({
                'crash_date': peak_date.strftime('%Y-%m-%d'), # 起跌日期
                'trough_date': crash['trough_date'].strftime('%Y-%m-%d'), # 最低點日期
                'depth_pct': f"{crash['depth']*100:.2f}%", # 跌幅
                'similarity': score, # 相似度
                'data': hist_slice # 用於繪圖的資料
            })
        except Exception as e:
            continue

    # 依照相似度排序 (高到低)
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    # -------------------------------------------------------------------------
    # 4. 介面輸出
    # -------------------------------------------------------------------------
    
    st.title(f"🚨 市場模式與崩盤預警: {symbol}")
    st.markdown(f"比較當前市場結構 (**{lookback_days} 天**, 截止日 {input_date}) 與 **歷史前 10 大崩盤模式** 的相似度。")

    col1, col2, col3 = st.columns(3)
    col1.metric("當前價格 (Current Price)", f"{current_pattern_df['Close'].iloc[-1]:.2f}")
    col2.metric("當前最大回撤 (Current MDD)", f"{current_pattern_df['Drawdown'].iloc[-1]*100:.2f}%")
    top_sim = f"{results[0]['similarity']:.1f}%" if results else "N/A"
    col3.metric("最高相似度 (Top Match)", top_sim)

    # 1. 當前走勢圖
    st.subheader("1. 當前市場情境 (Current Context)")
    fig_current = create_comprehensive_chart(current_pattern_df, "當前區間分析")
    st.plotly_chart(fig_current, use_container_width=True)

    # 2. 相似度列表
    st.subheader("2. 歷史相似度矩陣 (Similarity Matrix)")
    
    display_data = []
    for r in results:
        display_data.append({
            "崩盤起跌日 (Peak)": r['crash_date'],
            "波段最低點 (Bottom)": r['trough_date'],
            "總跌幅 (Drawdown)": r['depth_pct'],
            "相似度分數 (Similarity)": f"{r['similarity']:.1f}%"
        })
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)

    # 3. 最佳匹配視覺化
    st.subheader("3. 最佳匹配歷史場景")
    
    if results:
        top_match = results[0]
        st.markdown(f"### 🏆 相似度第一名: {top_match['crash_date']} (相似度: {top_match['similarity']:.1f}%)")
        st.caption(f"此模式隨後在 {top_match['trough_date']} 跌至最低點，跌幅達 {top_match['depth_pct']}。")
        
        fig_top = create_comprehensive_chart(top_match['data'], f"歷史回顧: {top_match['crash_date']}")
        st.plotly_chart(fig_top, use_container_width=True)
        
        # 其他匹配結果
        with st.expander("查看其他高相似度歷史場景"):
            for i, res in enumerate(results[1:], 1):
                st.markdown(f"#### 排名 {i+1}: {res['crash_date']} (相似度: {res['similarity']:.1f}%)")
                fig_hist = create_comprehensive_chart(res['data'], f"歷史回顧: {res['crash_date']}")
                st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{i}")

if __name__ == "__main__":
    main()