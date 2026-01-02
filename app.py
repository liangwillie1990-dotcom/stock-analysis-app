"""
Willie 戰情室 V14.1 - Performance Optimized Edition
Author: Gemini AI
Description: Fixed loading issues by implementing lightweight fetching for dashboard.
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

# ==========================================
# 0. 全局設定
# ==========================================
st.set_page_config(
    page_title="Willie 戰情室 V14.1",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root { --primary: #00d2ff; --bg: #0e1117; }
    .stApp { font-family: 'Microsoft JhengHei', sans-serif; background-color: var(--bg); }
    div[data-testid="stMetric"] {
        background-color: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 15px;
    }
    div[data-testid="stMetric"]:hover { border-color: var(--primary); }
    .stTabs [data-baseweb="tab-list"] { background-color: #0b0e14; padding: 10px; border-radius: 10px; }
    .stButton>button { background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%); color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫 (DB)
# ==========================================
DB_NAME = "willie_invest.db"

class DBManager:
    @staticmethod
    def init_db():
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS stock_cache (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY, date TIMESTAMP, ticker TEXT, type TEXT, price REAL, shares INTEGER, amount REAL, fee REAL, note TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, avg_cost REAL, shares INTEGER, group_name TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS system_config (key TEXT PRIMARY KEY, value TEXT)''')
            conn.commit()
            conn.close()
            DBManager.seed_data()
        except: pass

    @staticmethod
    def seed_data():
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

    @staticmethod
    def record_transaction(ticker, trans_type, price, shares):
        date = datetime.now()
        amount = price * shares
        fee = int(amount * 0.001425)
        tax = int(amount * 0.003) if trans_type == 'SELL' else 0
        total = amount + fee if trans_type == 'BUY' else amount - fee - tax
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''INSERT INTO transactions (date, ticker, type, price, shares, amount, fee, note)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (date, ticker, trans_type, price, shares, total, fee+tax, "User"))
        
        c.execute("SELECT avg_cost, shares FROM portfolio WHERE ticker=?", (ticker,))
        row = c.fetchone()
        
        if trans_type == 'BUY':
            if row:
                old_c, old_s = row
                new_s = old_s + shares
                new_c = ((old_c * old_s) + total) / new_s
                c.execute("UPDATE portfolio SET avg_cost=?, shares=? WHERE ticker=?", (new_c, new_s, ticker))
            else:
                c.execute("INSERT INTO portfolio (ticker, avg_cost, shares, group_name) VALUES (?, ?, ?, ?)", (ticker, total/shares, shares, 'Default'))
        elif trans_type == 'SELL' and row:
            old_c, old_s = row
            if shares >= old_s: c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
            else: c.execute("UPDATE portfolio SET shares=? WHERE ticker=?", (old_s - shares, ticker))
            
        conn.commit()
        conn.close()
        return f"交易成功: {trans_type} {ticker}"

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
# 2. 技術分析與風險引擎
# ==========================================
class TAEngine:
    @staticmethod
    def calculate(df):
        if df.empty: return df
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        # KD
        rsv = (df['Close'] - df['Low'].rolling(9).min()) / (df['High'].rolling(9).max() - df['Low'].rolling(9).min()) * 100
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # BB
        df['BB_Up'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
        df['BB_Low'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
        return df

class RiskEngine:
    @staticmethod
    def calculate_metrics(df):
        if len(df) < 30: return {}
        ret = df['Close'].pct_change()
        vol = ret.std() * np.sqrt(252)
        total_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        cagr = (1 + total_ret) ** (1/(len(df)/252)) - 1
        sharpe = (cagr - 0.015) / vol if vol != 0 else 0
        dd = (df['Close'] / df['Close'].cummax() - 1).min()
        return {"volatility": vol, "sharpe": sharpe, "max_dd": dd}

# ==========================================
# 3. 數據抓取引擎 (優化版)
# ==========================================
class DataFetcher:
    @staticmethod
    def normalize_ticker(ticker):
        ticker = ticker.strip().upper()
        if ticker.isdigit(): ticker += ".TW"
        return ticker

    @staticmethod
    def fetch_simple_quote(ticker):
        """極速模式：只抓價格與漲跌 (用於儀表板)"""
        # 1. Twstock (台股優先)
        if ticker[:2].isdigit():
            try:
                sid = ticker.replace(".TW", "")
                real = twstock.realtime.get(sid)
                if real['success']:
                    return {
                        "ticker": ticker,
                        "price": float(real['realtime']['latest_trade_price']),
                        "change_pct": 0.0, # Twstock 即時沒給漲跌幅，儀表板可接受暫無
                        "name": real['info']['name']
                    }
            except: pass
            
        # 2. Yahoo (指數/美股/或 Twstock 失敗)
        try:
            # 針對指數或期貨，不要加 .TW
            y_ticker = ticker
            stock = yf.Ticker(y_ticker)
            # 只抓 5 天，速度快 10 倍
            hist = stock.history(period="5d")
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                return {
                    "ticker": ticker,
                    "price": curr,
                    "change_pct": (curr - prev) / prev * 100,
                    "name": ticker
                }
        except: pass
        return None

    @staticmethod
    def fetch_full(ticker):
        """完整模式：抓全套數據 (用於個股分析)"""
        ticker = DataFetcher.normalize_ticker(ticker)
        cached = DBManager.get_cache(ticker)
        if cached: return cached
        
        data = {"ticker": ticker}
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty: return None
            
            # 補即時價
            if ticker[:2].isdigit():
                try:
                    real = twstock.realtime.get(ticker.replace(".TW", ""))
                    if real['success']: 
                        data['price'] = float(real['realtime']['latest_trade_price'])
                        data['name'] = real['info']['name']
                except: pass
            
            if 'price' not in data: 
                data['price'] = hist['Close'].iloc[-1]
                data['name'] = ticker

            hist = TAEngine.calculate(hist)
            info = stock.info
            eps = info.get('trailingEps')
            
            val = {}
            if eps:
                pe_s = hist['Close'] / eps
                val = {"cheap": eps*pe_s.min(), "fair": eps*pe_s.mean(), "expensive": eps*pe_s.max()}
            
            data.update({
                "change_pct": (data['price'] - hist['Close'].iloc[-2])/hist['Close'].iloc[-2]*100,
                "volume": hist['Volume'].iloc[-1],
                "pe": data['price']/eps if eps else None,
                "yield": info.get('dividendYield', 0)*100,
                "history_json": hist.reset_index().to_json(date_format='iso'),
                "valuation": val,
                "risk": RiskEngine.calculate_metrics(hist)
            })
            DBManager.save_cache(ticker, data)
            return data
        except: return None

    @staticmethod
    def fetch_batch_simple(tickers):
        """儀表板專用：並行極速抓取"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(DataFetcher.fetch_simple_quote, tickers))
        return [r for r in results if r]

    @staticmethod
    def fetch_batch_full(tickers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(DataFetcher.fetch_full, tickers))
        return [r for r in results if r]

# ==========================================
# 4. 背景排程
# ==========================================
def run_scheduler():
    while True:
        if datetime.now().strftime("%H:%M") == "07:30":
            df = DBManager.get_portfolio()
            if not df.empty: DataFetcher.fetch_batch_full(df['ticker'].tolist())
        time.sleep(60)

@st.cache_resource
def start_thread():
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    return t
start_thread()

# ==========================================
# 5. UI 與 主程式
# ==========================================
with st.sidebar:
    st.title("Willie 戰情室")
    st.info("V14.1 極速優化版")
    
    with st.expander("⚡ 快速下單 (Ledger)"):
        t_t = st.text_input("代號", "2330").upper()
        t_a = st.radio("動作", ["BUY", "SELL"], horizontal=True)
        t_p = st.number_input("價格", 0.0)
        t_s = st.number_input("股數", 1)
        if st.button("記錄"):
            msg = DBManager.record_transaction(DataFetcher.normalize_ticker(t_t), t_a, t_p, t_s)
            st.success(msg)
            time.sleep(1)
            st.rerun()

tabs = st.tabs(["📊 全球儀表板", "🔎 個股戰情室", "🎯 策略篩選", "💰 帳本與損益"])

# Tab 1: 儀表板 (使用 simple_fetch 解決 Loading 問題)
with tabs[0]:
    st.subheader("🌍 全球市場概況 (即時)")
    items = {"^TWII":"加權指數", "^TWOII":"櫃買指數", "^SOX":"費半", "^IXIC":"那指", "GC=F":"黃金", "SI=F":"白銀", "CL=F":"原油", "USDTWD=X":"匯率"}
    
    # 使用優化過的簡單抓取
    data_list = DataFetcher.fetch_batch_simple(list(items.keys()))
    
    cols = st.columns(4)
    for i, (k, v) in enumerate(items.items()):
        d = next((x for x in data_list if x['ticker'] == k), None)
        with cols[i % 4]:
            if d: st.metric(v, f"{d['price']:,.2f}", f"{d['change_pct']:.2f}%")
            else: st.metric(v, "N/A", "查無資料")
        if (i+1) % 4 == 0: st.write("")

# Tab 2: 個股 (使用 full_fetch)
with tabs[1]:
    col1, col2 = st.columns([3, 1])
    target = col1.text_input("輸入代號", "2330.TW").upper()
    if col2.button("深度分析"):
        DBManager.save_cache(DataFetcher.normalize_ticker(target), {})
    
    d = DataFetcher.fetch_full(target)
    if d:
        st.markdown(f"### {d.get('name', target)}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("現價", d['price'], f"{d['change_pct']:.2f}%")
        c2.metric("PE", f"{d['pe']:.1f}x" if d['pe'] else "-")
        c3.metric("殖利率", f"{d['yield']:.2f}%")
        c4.metric("夏普", f"{d['risk']['sharpe']:.2f}")
        c5.metric("波動", f"{d['risk']['volatility']*100:.1f}%")
        
        # K線圖
        df = pd.read_json(d['history_json'])
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']); df.set_index('Date', inplace=True)
        elif 'index' in df.columns: df['index'] = pd.to_datetime(df['index']); df.set_index('index', inplace=True)
        
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='MA20'))
        fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # 估值
        if d.get('valuation'):
            val = d['valuation']
            st.info(f"💎 估價區間： 便宜 {val['cheap']:.1f}  |  合理 {val['fair']:.1f}  |  昂貴 {val['expensive']:.1f}")

# Tab 3: 篩選
with tabs[2]:
    st.subheader("🎯 庫存健檢篩選")
    df_p = DBManager.get_portfolio()
    if not df_p.empty:
        if st.button("掃描庫存"):
            res = DataFetcher.fetch_batch_full(df_p['ticker'].tolist())
            rows = []
            for r in res:
                rows.append({"代號": r['ticker'], "現價": r['price'], "PE": r['pe'], "殖利率": r['yield'], "夏普": r['risk']['sharpe']})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else: st.warning("請先新增庫存")

# Tab 4: 帳本
with tabs[3]:
    st.subheader("💰 損益表")
    df_p = DBManager.get_portfolio()
    if not df_p.empty:
        tickers = df_p['ticker'].tolist()
        updates = DataFetcher.fetch_batch_simple(tickers) # 用快速模式抓現價
        price_map = {u['ticker']: u['price'] for u in updates}
        
        rows = []
        tm, tc = 0, 0
        for _, r in df_p.iterrows():
            curr = price_map.get(r['ticker'], r['avg_cost'])
            mkt = curr * r['shares']
            cost = r['avg_cost'] * r['shares']
            tm += mkt; tc += cost
            rows.append({"代號": r['ticker'], "股數": r['shares'], "成本": r['avg_cost'], "現價": curr, "損益": int(mkt-cost)})
        
        c1, c2 = st.columns(2)
        c1.metric("總市值", f"${tm:,.0f}")
        c2.metric("總損益", f"${tm-tc:,.0f}", f"{(tm-tc)/tc*100:.2f}%")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    st.subheader("交易紀錄")
    st.dataframe(DBManager.get_transactions(), use_container_width=True)
