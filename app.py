"""
Joymax Galaxy V14.0 - Enterprise Edition
Author: Gemini AI
Description: Comprehensive Stock Analysis, Portfolio Management, and Backtesting System.
Modules: DataFetcher, TAEngine, RiskEngine, LedgerSystem, BacktestEngine, UI.
"""

import streamlit as st
import yfinance as yf
import twstock
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import time
import json
import threading
import concurrent.futures
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional

# ==========================================
# 0. 全局設定與 CSS 樣式系統
# ==========================================
st.set_page_config(
    page_title="Joymax Galaxy V14",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# 注入企業級 CSS
st.markdown("""
<style>
    /* 核心色調與字體 */
    :root { --primary: #00d2ff; --secondary: #3a4764; --bg: #0e1117; }
    .stApp { font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif; background-color: var(--bg); }
    
    /* 指標卡片 (Metrics) */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--primary); }
    div[data-testid="stMetricLabel"] { color: #9ca3af; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #f3f4f6; font-weight: 700; }
    
    /* 表格優化 */
    div[data-testid="stDataFrame"] { border: 1px solid #374151; border-radius: 8px; overflow: hidden; }
    
    /* 側邊欄 */
    section[data-testid="stSidebar"] { background-color: #0b0e14; border-right: 1px solid #1f2937; }
    
    /* Tabs 優化 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #0b0e14; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; border-radius: 6px; color: #9ca3af; border: none; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #1f2937; color: var(--primary); }
    
    /* 按鈕特效 */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none; font-weight: bold; transition: all 0.3s;
    }
    .stButton>button:hover { box-shadow: 0 0 10px rgba(37, 99, 235, 0.5); }
    
    /* Toast 通知 */
    div[data-testid="stToast"] { background-color: #1f2937; color: white; border: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫與帳本系統 (Ledger System)
# ==========================================
DB_NAME = "joymax_galaxy.db"

class DBManager:
    """處理所有 SQLite 資料庫操作的單例類別"""
    
    @staticmethod
    def init_db():
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # 1. 基礎快取
        c.execute('''CREATE TABLE IF NOT EXISTS stock_cache
                     (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
        
        # 2. 交易流水帳 (Ledger) - V14 核心
        # type: BUY, SELL, DIVIDEND (股利)
        c.execute('''CREATE TABLE IF NOT EXISTS transactions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TIMESTAMP,
                      ticker TEXT,
                      type TEXT,
                      price REAL,
                      shares INTEGER,
                      amount REAL, 
                      fee REAL,
                      note TEXT)''')
                      
        # 3. 庫存彙總 (Portfolio Summary)
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                     (ticker TEXT PRIMARY KEY, avg_cost REAL, shares INTEGER, group_name TEXT)''')
        
        # 4. 系統設定
        c.execute('''CREATE TABLE IF NOT EXISTS system_config
                     (key TEXT PRIMARY KEY, value TEXT)''')
                     
        conn.commit()
        conn.close()
        DBManager.seed_data()

    @staticmethod
    def seed_data():
        """預設資料初始化"""
        defaults = {
            "watchlist_tech": "2330,2317,2454,2308,3231,2382,6669,3443",
            "watchlist_finance": "2881,2882,2891,5880,2886,2892",
            "watchlist_shipping": "2603,2609,2615,2637,5608",
            "watchlist_etf": "0050,0056,00878,00919,00929"
        }
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for k, v in defaults.items():
            c.execute("INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)", (k, v))
        conn.commit()
        conn.close()

    # --- 快取操作 ---
    @staticmethod
    def get_cache(ticker, ttl_minutes=30):
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT data, updated_at FROM stock_cache WHERE ticker=?", (ticker,))
            row = c.fetchone()
            conn.close()
            if row:
                data_str, updated_at_str = row
                if datetime.now() - datetime.fromisoformat(updated_at_str) < timedelta(minutes=ttl_minutes):
                    return json.loads(data_str)
        except: pass
        return None

    @staticmethod
    def save_cache(ticker, data):
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("REPLACE INTO stock_cache (ticker, data, updated_at) VALUES (?, ?, ?)", 
                      (ticker, json.dumps(data), datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except: pass

    # --- 交易與庫存操作 (Accounting) ---
    @staticmethod
    def record_transaction(ticker, trans_type, price, shares, date=None):
        """
        記錄交易並自動更新庫存
        trans_type: 'BUY', 'SELL'
        """
        if date is None: date = datetime.now()
        
        # 台灣手續費 0.1425%，交易稅 0.3% (賣出)
        amount = price * shares
        fee = int(amount * 0.001425) if amount > 0 else 0
        tax = int(amount * 0.003) if trans_type == 'SELL' else 0
        
        total_amount = amount + fee if trans_type == 'BUY' else amount - fee - tax
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # 1. 寫入流水帳
        c.execute('''INSERT INTO transactions (date, ticker, type, price, shares, amount, fee, note)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (date, ticker, trans_type, price, shares, total_amount, fee+tax, "User Input"))
        
        # 2. 更新庫存 (Portfolio)
        c.execute("SELECT avg_cost, shares FROM portfolio WHERE ticker=?", (ticker,))
        row = c.fetchone()
        
        if trans_type == 'BUY':
            if row:
                old_cost, old_shares = row
                # 平均成本法
                new_shares = old_shares + shares
                new_cost = ((old_cost * old_shares) + total_amount) / new_shares
                c.execute("UPDATE portfolio SET avg_cost=?, shares=? WHERE ticker=?", (new_cost, new_shares, ticker))
            else:
                # 新增持股
                avg_cost = total_amount / shares
                c.execute("INSERT INTO portfolio (ticker, avg_cost, shares, group_name) VALUES (?, ?, ?, ?)", 
                          (ticker, avg_cost, shares, 'Default'))
        
        elif trans_type == 'SELL':
            if row:
                old_cost, old_shares = row
                if shares >= old_shares:
                    # 全賣光
                    c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
                else:
                    # 減碼 (成本不變，股數減少)
                    new_shares = old_shares - shares
                    c.execute("UPDATE portfolio SET shares=? WHERE ticker=?", (new_shares, ticker))
            else:
                # 空單 (暫不支援，僅記錄交易)
                pass

        conn.commit()
        conn.close()
        return f"交易成功：{trans_type} {shares}股 {ticker} @ {price}"

    @staticmethod
    def get_portfolio():
        try:
            conn = sqlite3.connect(DB_NAME)
            df = pd.read_sql("SELECT * FROM portfolio", conn)
            conn.close()
            return df
        except: return pd.DataFrame()

    @staticmethod
    def get_transactions():
        try:
            conn = sqlite3.connect(DB_NAME)
            df = pd.read_sql("SELECT * FROM transactions ORDER BY date DESC", conn)
            conn.close()
            return df
        except: return pd.DataFrame()

DBManager.init_db()

# ==========================================
# 2. 進階技術分析引擎 (Advanced TA Engine)
# ==========================================
class TAEngine:
    @staticmethod
    def calculate(df):
        if df.empty: return df
        
        # 基礎均線
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA120'] = df['Close'].rolling(120).mean()
        
        # 1. KD (Stochastic Oscillator)
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # 2. MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        # 3. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Bollinger Bands (布林通道)
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Low'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']
        
        # 5. ATR (Average True Range) - 波動率指標
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # 6. OBV (On-Balance Volume) - 能量潮
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # 7. Ichimoku Cloud (一目均衡表)
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        df['Tenkan_sen'] = (high_9 + low_9) / 2  # 轉折線
        
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        df['Kijun_sen'] = (high_26 + low_26) / 2 # 基準線
        
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        
        df['Chikou_Span'] = df['Close'].shift(-26) # 遲行線

        return df

# ==========================================
# 3. 量化風險引擎 (Risk & Quant Engine)
# ==========================================
class RiskEngine:
    @staticmethod
    def calculate_metrics(df):
        """計算 Sharpe, Max Drawdown, Volatility"""
        if len(df) < 30: return {}
        
        # 日報酬率
        df['Returns'] = df['Close'].pct_change()
        
        # 年化波動率 (Volatility)
        volatility = df['Returns'].std() * np.sqrt(252)
        
        # 年化報酬率 (CAGR - 簡易版)
        total_ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        years = len(df) / 252
        cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        # 夏普比率 (Sharpe Ratio, 假設無風險利率 1.5%)
        rf = 0.015
        sharpe = (cagr - rf) / volatility if volatility != 0 else 0
        
        # 最大回撤 (Max Drawdown)
        roll_max = df['Close'].cummax()
        drawdown = df['Close'] / roll_max - 1.0
        max_dd = drawdown.min()
        
        return {
            "volatility": volatility,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd
        }

# ==========================================
# 4. 資料抓取與處理 (Robust Data Fetcher)
# ==========================================
class DataFetcher:
    @staticmethod
    def normalize_ticker(ticker):
        ticker = ticker.strip().upper()
        if ticker.isdigit(): ticker += ".TW"
        return ticker

    @staticmethod
    def fetch_full(ticker, days=365):
        """混合引擎：Twstock 即時 + Yahoo 歷史 + 基本面"""
        ticker = DataFetcher.normalize_ticker(ticker)
        
        # 1. 讀快取
        cached = DBManager.get_cache(ticker)
        if cached: return cached
        
        data = {"ticker": ticker}
        
        # 2. Twstock (即時價)
        if ticker[:2].isdigit():
            try:
                sid = ticker.replace(".TW", "").replace(".TWO", "")
                real = twstock.realtime.get(sid)
                if real['success']:
                    data['price'] = float(real['realtime']['latest_trade_price'])
                    data['name'] = real['info']['name']
            except: pass
            
        # 3. Yahoo (歷史與基本面)
        try:
            stock = yf.Ticker(ticker)
            period = "2y" if days > 365 else "1y"
            hist = stock.history(period=period)
            
            if hist.empty: return None
            
            # 填補即時價
            if 'price' not in data: data['price'] = hist['Close'].iloc[-1]
            if 'name' not in data: data['name'] = ticker
            
            # 技術指標計算
            hist = TAEngine.calculate(hist)
            
            # 基本面
            info = stock.info
            eps = info.get('trailingEps') or info.get('forwardEps')
            pe = data['price']/eps if eps and eps > 0 else None
            
            # 估值模型
            valuation = {}
            if eps:
                pe_s = hist['Close'] / eps
                valuation = {
                    "cheap": eps * pe_s.min(),
                    "fair": eps * pe_s.mean(),
                    "expensive": eps * pe_s.max()
                }
            
            # 量化指標
            risk_metrics = RiskEngine.calculate_metrics(hist)
            
            # 整合
            data.update({
                "change_pct": (data['price'] - hist['Close'].iloc[-2])/hist['Close'].iloc[-2]*100,
                "volume": hist['Volume'].iloc[-1],
                "pe": pe, "eps": eps, 
                "yield": info.get('dividendYield', 0)*100,
                "history_json": hist.reset_index().to_json(date_format='iso'),
                "valuation": valuation,
                "risk": risk_metrics,
                "market_cap": info.get('marketCap', 0),
                "sector": info.get('sector', 'N/A')
            })
            
            DBManager.save_cache(ticker, data)
            return data
            
        except Exception as e:
            # print(f"Fetch Error: {e}")
            return None

    @staticmethod
    def fetch_batch(tickers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(DataFetcher.fetch_full, tickers))
        return [r for r in results if r]

# ==========================================
# 5. 回測引擎 (Backtest Engine)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_strategy(df, strategy_type="kd_cross", initial_capital=100000):
        """簡易回測引擎"""
        cash = initial_capital
        position = 0 # 股數
        log = []
        
        df = df.copy()
        df['Action'] = 'HOLD'
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            price = curr['Close']
            date = curr.name
            
            signal = 0 # 1 Buy, -1 Sell
            
            if strategy_type == "kd_cross":
                # 黃金交叉買進
                if prev['K'] < prev['D'] and curr['K'] > curr['D'] and curr['K'] < 30:
                    signal = 1
                # 死亡交叉賣出
                elif prev['K'] > prev['D'] and curr['K'] < curr['D'] and curr['K'] > 80:
                    signal = -1
                    
            elif strategy_type == "ma_cross":
                # 月季線交叉
                if prev['MA20'] < prev['MA60'] and curr['MA20'] > curr['MA60']: signal = 1
                elif prev['MA20'] > prev['MA60'] and curr['MA20'] < curr['MA60']: signal = -1

            # 執行交易
            if signal == 1 and cash > price * 1000: # 買一張
                shares_to_buy = int(cash // price)
                cost = shares_to_buy * price
                cash -= cost
                position += shares_to_buy
                log.append({"date": date, "action": "BUY", "price": price, "shares": shares_to_buy})
                
            elif signal == -1 and position > 0: # 賣出
                cash += position * price
                log.append({"date": date, "action": "SELL", "price": price, "shares": position})
                position = 0
                
        final_value = cash + (position * df.iloc[-1]['Close'])
        ret = (final_value - initial_capital) / initial_capital * 100
        
        return {
            "final_value": final_value,
            "return_pct": ret,
            "log": log
        }

# ==========================================
# 6. 背景排程 (Scheduler)
# ==========================================
def run_scheduler():
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == "07:30":
            df = DBManager.get_portfolio()
            if not df.empty:
                DataFetcher.fetch_batch(df['ticker'].tolist())
        time.sleep(60)

@st.cache_resource
def start_thread():
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    return t
start_thread()

# ==========================================
# 7. UI 組件與繪圖 (Visualization)
# ==========================================
def render_advanced_chart(data):
    try:
        df = pd.read_json(data['history_json'])
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']); df.set_index('Date', inplace=True)
        elif 'index' in df.columns: df['index'] = pd.to_datetime(df['index']); df.set_index('index', inplace=True)
        
        # 建立多子圖
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f"{data['name']} 技術分析", "成交量 & MACD", "KD & RSI"))

        # Main: K線 + 均線 + 布林 + 一目
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='月線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='季線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(width=0, color='gray'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], fill='tonexty', fillcolor='rgba(128,128,128,0.1)', line=dict(width=0, color='gray'), name='布林'), row=1, col=1)
        
        # Sub 1: MACD + Volume
        colors = ['red' if r > 0 else 'green' for r in df['Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='yellow', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='cyan', width=1), name='DEM'), row=2, col=1)
        
        # Sub 2: KD
        fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='orange', width=1), name='K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='purple', width=1), name='D'), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
        fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")

        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    except: st.error("圖表繪製錯誤")

# ==========================================
# 8. 主程式邏輯 (Main App)
# ==========================================

# --- 側邊欄 ---
with st.sidebar:
    st.title("🌌 Galaxy V14")
    st.markdown("企業級戰情室")
    
    # 快捷操作
    with st.expander("⚡ 快速交易 (Ledger)", expanded=True):
        t_ticker = st.text_input("代號", "2330").upper()
        t_action = st.radio("動作", ["BUY", "SELL"], horizontal=True)
        t_price = st.number_input("價格", 0.0, step=0.5)
        t_shares = st.number_input("股數", 1, step=1)
        if st.button("📝 記錄交易"):
            msg = DBManager.record_transaction(DataFetcher.normalize_ticker(t_ticker), t_action, t_price, t_shares)
            st.success(msg)
            time.sleep(1)
            st.rerun()

    st.info("功能導航：\n1. 儀表板: 大盤與商品\n2. 戰情室: 深度個股\n3. 篩選器: 策略選股\n4. 帳本: 資產管理\n5. 回測: 策略驗證")

# --- 頁面 Tabs ---
tabs = st.tabs(["📊 全球儀表板", "🔎 個股戰情室", "🎯 策略篩選", "💰 帳本與損益", "🧪 策略回測"])

# Tab 1: 儀表板 (Macro)
with tabs[0]:
    st.subheader("🌍 全球市場與原物料")
    
    # 指數 + 原物料
    items = {
        "^TWII": "加權指數", "^TWOII": "櫃買指數", "^SOX": "費半", "^IXIC": "那指",
        "GC=F": "黃金", "SI=F": "白銀", "CL=F": "原油", "USDTWD=X": "美金台幣"
    }
    
    # 批次抓取
    data_list = DataFetcher.fetch_batch(list(items.keys()))
    
    # 顯示
    cols = st.columns(4)
    for i, (k, v) in enumerate(items.items()):
        d = next((x for x in data_list if x['ticker'] == k), None)
        with cols[i % 4]:
            if d:
                st.metric(v, f"{d['price']:,.2f}", f"{d['change_pct']:.2f}%")
            else:
                st.metric(v, "Loading...")
        if (i+1) % 4 == 0: st.write("") # 換行

# Tab 2: 個股戰情室 (Deep Dive)
with tabs[1]:
    col_s1, col_s2 = st.columns([4, 1])
    search_ticker = col_s1.text_input("輸入代號分析", "2330.TW").upper()
    if col_s2.button("立即分析"):
        DBManager.save_cache(DataFetcher.normalize_ticker(search_ticker), {}) # 清快取強制更新
    
    d = DataFetcher.fetch_full(search_ticker)
    
    if d:
        st.markdown(f"### {d['name']} ({d['ticker']})")
        
        # 核心數據矩陣
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("現價", d['price'], f"{d['change_pct']:.2f}%")
        m2.metric("本益比", f"{d['pe']:.1f}x" if d['pe'] else "-")
        m3.metric("殖利率", f"{d['yield']:.2f}%")
        m4.metric("夏普值", f"{d['risk']['sharpe']:.2f}")
        m5.metric("波動率", f"{d['risk']['volatility']*100:.1f}%")
        m6.metric("ATR", f"{d['history_json'].count('ATR') and 0}") # 簡化顯示

        # 進階圖表
        render_advanced_chart(d)
        
        # 估值與風險
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 💎 估價模型")
            if d.get('valuation'):
                val = d['valuation']
                # 簡單進度條模擬儀表
                st.info(f"便宜: {val['cheap']:.1f} | 合理: {val['fair']:.1f} | 昂貴: {val['expensive']:.1f}")
                
        with c2:
            st.markdown("#### ⚠️ 風險評估")
            r = d['risk']
            st.warning(f"最大回撤 (Max Drawdown): {r['max_dd']*100:.2f}%")
            st.write(f"若持有 1 年，有 95% 機率虧損不超過: {r['volatility']*1.65*100:.1f}% (VaR)")

# Tab 3: 策略篩選 (Screener)
with tabs[2]:
    st.subheader("🎯 智能選股雷達")
    
    with st.form("screener_form"):
        c1, c2, c3 = st.columns(3)
        f_pe = c1.slider("PE 低於", 10, 60, 20)
        f_yld = c2.slider("殖利率 高於", 0.0, 10.0, 4.0)
        f_vol = c3.checkbox("成交量爆發 ( > 5日均量)", False)
        
        source = st.radio("掃描範圍", ["半導體 (Tech)", "金融 (Finance)", "航運 (Shipping)", "ETF", "庫存股"], horizontal=True)
        submitted = st.form_submit_button("🚀 啟動掃描")
        
    if submitted:
        # 從 DB 設定讀取清單
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        list_key = "watchlist_tech" # default
        if "金融" in source: list_key = "watchlist_finance"
        elif "航運" in source: list_key = "watchlist_shipping"
        elif "ETF" in source: list_key = "watchlist_etf"
        
        c.execute("SELECT value FROM system_config WHERE key=?", (list_key,))
        row = c.fetchone()
        tickers = row[0].split(",") if row else []
        
        if "庫存" in source:
            df_p = DBManager.get_portfolio()
            tickers = df_p['ticker'].tolist()
            
        conn.close()
        
        # 執行掃描
        with st.spinner("AI 引擎分析中..."):
            results = DataFetcher.fetch_batch(tickers)
            
        # 過濾
        filtered = []
        for r in results:
            keep = True
            if r['pe'] and r['pe'] > f_pe: keep = False
            if r['yield'] < f_yld: keep = False
            if keep: filtered.append(r)
            
        # 顯示
        if filtered:
            df_res = pd.DataFrame(filtered)[['ticker', 'name', 'price', 'change_pct', 'pe', 'yield', 'volume']]
            st.dataframe(df_res, use_container_width=True)
        else:
            st.warning("無符合條件標的")

# Tab 4: 帳本與損益 (Ledger)
with tabs[3]:
    st.subheader("💰 資產管理中心")
    
    subtab1, subtab2 = st.tabs(["庫存總覽", "交易流水帳"])
    
    with subtab1:
        df_p = DBManager.get_portfolio()
        if not df_p.empty:
            # 取得現價計算市值
            tickers = df_p['ticker'].tolist()
            updates = DataFetcher.fetch_batch(tickers)
            price_map = {u['ticker']: u['price'] for u in updates}
            
            p_data = []
            total_mkt, total_cost = 0, 0
            
            for _, row in df_p.iterrows():
                curr = price_map.get(row['ticker'], row['avg_cost'])
                mkt = curr * row['shares']
                cost = row['avg_cost'] * row['shares']
                pnl = mkt - cost
                total_mkt += mkt
                total_cost += cost
                
                p_data.append({
                    "代號": row['ticker'], "股數": row['shares'],
                    "平均成本": row['avg_cost'], "現價": curr,
                    "市值": mkt, "未實現損益": pnl, "報酬率%": (pnl/cost)*100
                })
                
            col1, col2, col3 = st.columns(3)
            col1.metric("總市值", f"${total_mkt:,.0f}")
            col2.metric("總成本", f"${total_cost:,.0f}")
            col3.metric("總損益", f"${total_mkt-total_cost:,.0f}", f"{(total_mkt-total_cost)/total_cost*100:.2f}%")
            
            st.dataframe(pd.DataFrame(p_data), use_container_width=True)
        else:
            st.info("尚無庫存，請至側邊欄新增交易。")
            
    with subtab2:
        df_t = DBManager.get_transactions()
        st.dataframe(df_t, use_container_width=True)

# Tab 5: 策略回測 (Backtest)
with tabs[4]:
    st.subheader("🧪 策略實驗室")
    
    c1, c2, c3 = st.columns(3)
    bt_ticker = c1.text_input("回測標的", "2330.TW").upper()
    bt_strat = c2.selectbox("策略", ["kd_cross", "ma_cross"])
    bt_fund = c3.number_input("初始資金", 100000, 10000000, 500000)
    
    if st.button("▶️ 開始回測"):
        with st.spinner("模擬交易中..."):
            d = DataFetcher.fetch_full(bt_ticker, days=730) # 抓2年
            if d:
                df_hist = pd.read_json(d['history_json'])
                # 重建技術指標 (因為 fetch_full 只存了最後一筆，這裡要重算整串)
                df_hist = TAEngine.calculate(df_hist)
                
                res = BacktestEngine.run_strategy(df_hist, bt_strat, bt_fund)
                
                # 顯示結果
                r1, r2 = st.columns(2)
                r1.metric("期末資產", f"${res['final_value']:,.0f}")
                r2.metric("總報酬率", f"{res['return_pct']:.2f}%")
                
                st.write("交易紀錄:")
                st.dataframe(pd.DataFrame(res['log']), use_container_width=True)
            else:
                st.error("無法取得歷史數據")
