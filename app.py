"""
Willie's Omega V21.0 - Genesis (Back to Basics)
Author: Gemini AI
Description:
    Removes all complex fallback/spoofing mechanisms.
    Focuses on strict data validation to prevent KeyErrors.
    Uses pure yfinance with standard error handling.
"""

import streamlit as st
import yfinance as yf
import twstock
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import datetime

# ==========================================
# 0. 全局設定
# ==========================================
st.set_page_config(
    page_title="Willie's Omega V21",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root { --primary: #00d4ff; --bg: #0e1117; }
    .stApp { font-family: 'Microsoft JhengHei', sans-serif; background-color: var(--bg); }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; border-radius: 8px; padding: 10px; }
    .stButton>button { background: #00d4ff; color: black; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫 (最簡化)
# ==========================================
DB_NAME = "willie_v21.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, cost REAL, shares INTEGER)''')
    conn.commit()
    conn.close()

def get_portfolio():
    conn = sqlite3.connect(DB_NAME)
    try:
        return pd.read_sql("SELECT * FROM portfolio", conn)
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def update_portfolio(ticker, cost, shares):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 簡單邏輯：有就覆蓋，沒有就新增
    c.execute("REPLACE INTO portfolio (ticker, cost, shares) VALUES (?, ?, ?)", (ticker, cost, shares))
    conn.commit()
    conn.close()

def delete_portfolio(ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. 數據核心 (嚴格驗證版)
# ==========================================
class DataCore:
    @staticmethod
    def normalize(ticker):
        ticker = ticker.strip().upper()
        if ticker.isdigit(): ticker += ".TW"
        return ticker

    @staticmethod
    def fetch_data(ticker):
        ticker = DataCore.normalize(ticker)
        
        # 1. 抓取數據
        try:
            stock = yf.Ticker(ticker)
            # 嘗試抓取，如果 Yahoo 擋 IP，這裡可能會回傳空 DataFrame
            hist = stock.history(period="1y")
        except Exception as e:
            st.error(f"連線失敗: {e}")
            return None

        # 2. 【關鍵】嚴格檢查數據有效性
        if hist.empty:
            return None
        
        # 檢查必要欄位是否存在 (這就是 V20 報錯的原因)
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        # 有些情況下 Volume 會丟失，我們至少要 Close
        if 'Close' not in hist.columns:
            st.warning(f"數據源異常，{ticker} 缺少收盤價欄位。")
            return None

        # 3. 補即時價 (Twstock 作為輔助，不強求)
        current_price = hist['Close'].iloc[-1]
        try:
            if ticker[:2].isdigit():
                # 這裡只抓 realtime，不抓 history，因為 history 容易 SSL 報錯
                real = twstock.realtime.get(ticker.replace(".TW", ""))
                if real['success']:
                    current_price = float(real['realtime']['latest_trade_price'])
        except: pass

        # 4. 整理回傳包
        # 確保有足夠數據算漲跌幅
        if len(hist) > 2:
            prev_close = hist['Close'].iloc[-2]
            change_pct = (current_price - prev_close) / prev_close * 100
        else:
            change_pct = 0.0

        return {
            "ticker": ticker,
            "price": current_price,
            "hist": hist,
            "info": stock.info,
            "change_pct": change_pct
        }

# ==========================================
# 3. 運算引擎 (防呆版)
# ==========================================
def calculate_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    
    try:
        # MA
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        # KD
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # MACD
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
    except Exception as e:
        print(f"指標計算錯誤: {e}")
        
    return df

def ai_score(df, info):
    if df is None or df.empty: return 0
    score = 60 # 基礎分
    
    try:
        curr = df.iloc[-1]
        
        # 技術面
        if 'MA20' in curr and curr['Close'] > curr['MA20']: score += 10
        if 'MA60' in curr and curr['Close'] > curr['MA60']: score += 10
        if 'K' in curr and curr['K'] > curr['D']: score += 5
        
        # 基本面 (防呆)
        pe = info.get('trailingPE')
        if pe and pe < 15: score += 10
        elif pe and pe > 30: score -= 10
        
        pb = info.get('priceToBook')
        if pb and pb < 1.5: score += 5
        
    except: pass
    
    return min(100, max(0, score))

# ==========================================
# 4. UI 介面
# ==========================================
with st.sidebar:
    st.title("🌱 Willie's V21")
    st.caption("Genesis | 原始穩定版")
    if st.button("清除快取"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("新增庫存")
    pt_t = st.text_input("代號", "2330").upper()
    pt_c = st.number_input("成本", 0.0)
    pt_s = st.number_input("股數", 1000)
    if st.button("儲存"):
        t_fmt = DataCore.normalize(pt_t)
        update_portfolio(t_fmt, pt_c, pt_s)
        st.success("已儲存")
        st.rerun()

tabs = st.tabs(["📊 個股分析", "💰 我的庫存"])

# --- Tab 1: 個股分析 ---
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    target = col1.text_input("輸入代號查詢", "2330.TW")
    if col2.button("查詢"):
        pass # 觸發重跑

    # 執行查詢
    data = DataCore.fetch_data(target)
    
    if data:
        # 計算指標
        df = calculate_indicators(data['hist'])
        score = ai_score(df, data['info'])
        
        # 顯示數據
        st.subheader(f"{data['ticker']} - AI 評分: {score}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("現價", f"{data['price']}", f"{data['change_pct']:.2f}%")
        m2.metric("PE (本益比)", f"{data['info'].get('trailingPE', 'N/A')}")
        
        # 安全顯示 KD/MACD
        k_val = f"{df['K'].iloc[-1]:.1f}" if 'K' in df.columns else "-"
        d_val = f"{df['D'].iloc[-1]:.1f}" if 'D' in df.columns else "-"
        m3.metric("KD (K/D)", f"{k_val} / {d_val}")
        
        macd_val = f"{df['Hist'].iloc[-1]:.2f}" if 'Hist' in df.columns else "-"
        m4.metric("MACD", macd_val)
        
        # 繪圖
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        
        # K線
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='MA20'), row=1, col=1)
        if 'MA60' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue'), name='MA60'), row=1, col=1)
        
        # 量
        if 'Volume' in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量'), row=2, col=1)
        
        fig.update_layout(height=500, template='plotly_dark', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning(f"找不到 {target} 的資料，或是 Yahoo Finance 暫時封鎖了連線。請稍後再試。")

# --- Tab 2: 庫存 ---
with tabs[1]:
    df_port = get_portfolio()
    if not df_port.empty:
        # 簡單計算市值
        res = []
        for idx, row in df_port.iterrows():
            d = DataCore.fetch_data(row['ticker'])
            curr = d['price'] if d else row['cost'] # 抓不到就用成本價顯示
            val = curr * row['shares']
            cost = row['cost'] * row['shares']
            res.append({
                "代號": row['ticker'],
                "成本": row['cost'],
                "現價": curr,
                "股數": row['shares'],
                "市值": int(val),
                "損益": int(val - cost),
                "報酬率%": round((val-cost)/cost*100, 2)
            })
        
        st.dataframe(pd.DataFrame(res), use_container_width=True)
        
        # 刪除功能
        del_t = st.selectbox("刪除代號", df_port['ticker'])
        if st.button("確認刪除"):
            delete_portfolio(del_t)
            st.rerun()
            
    else:
        st.info("目前沒有庫存資料")
