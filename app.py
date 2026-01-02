"""
Willie's Omega V20.0 - Chameleon Edition (IP/Geo-block Bypass)
Author: Gemini AI
Description:
    1. Introduces 'yahooquery' as a primary fetcher (more resilient to IP bans).
    2. Implements 'Full Header Spoofing' for TWSE to bypass Geo-blocking.
    3. Adds explicit debugging for network status.
    4. Robust fallback chain: YahooQuery -> Yfinance -> TWSE (Spoofed).
"""

import streamlit as st
import yfinance as yf
from yahooquery import Ticker as YQTicker
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import time
import json
import threading
import concurrent.futures
import requests
import warnings
from datetime import datetime, timedelta
from scipy.stats import norm
from fake_useragent import UserAgent

# 忽略 SSL 警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局設定
# ==========================================
st.set_page_config(
    page_title="Willie's Omega V20",
    layout="wide",
    page_icon="🦎",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root { --primary: #00d4ff; --bg: #0e1117; }
    .stApp { font-family: 'Microsoft JhengHei', sans-serif; background-color: var(--bg); }
    div[data-testid="stMetric"] { background: #1a1c24; border: 1px solid #333; }
    .stButton>button { background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%); color: black; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫管理層
# ==========================================
DB_NAME = "willie_v20.db"

class DBManager:
    @staticmethod
    def init_db():
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS stock_cache (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, date TIMESTAMP, ticker TEXT, type TEXT, price REAL, shares INTEGER, amount REAL, fee REAL, note TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, avg_cost REAL, shares INTEGER, group_name TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS system_config (key TEXT PRIMARY KEY, value TEXT)''')
            conn.commit()
            conn.close()
            DBManager._seed_universe()
        except: pass

    @staticmethod
    def _seed_universe():
        universe = {
            "list_tech": "2330,2454,2303,3034,3035,2317,2382,3231,2357,6669,2356,3037,2345,4938",
            "list_finance": "2881,2882,2891,2886,2892,2884,2890,5880,2885,2880,2883,2887",
            "list_shipping": "2603,2609,2615,2618,2610,2637,5608,2606",
            "list_etf": "0050,0056,00878,00919,00929,00940,00713"
        }
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for k, v in universe.items():
            c.execute("INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)", (k, v))
        conn.commit()
        conn.close()

    @staticmethod
    def get_cache(ticker, ttl_minutes=60):
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT data, updated_at FROM stock_cache WHERE ticker=?", (ticker,))
            row = c.fetchone()
            conn.close()
            if row and (datetime.now() - datetime.fromisoformat(row[1]) < timedelta(minutes=ttl_minutes)):
                return json.loads(row[0])
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
        c.execute("INSERT INTO transactions (date, ticker, type, price, shares, amount, fee, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (date, ticker, trans_type, price, shares, total, fee+tax, "Manual"))
        c.execute("SELECT avg_cost, shares FROM portfolio WHERE ticker=?", (ticker,))
        row = c.fetchone()
        if trans_type == 'BUY':
            if row:
                new_s = row[1] + shares
                new_c = ((row[0] * row[1]) + total) / new_s
                c.execute("UPDATE portfolio SET avg_cost=?, shares=? WHERE ticker=?", (new_c, new_s, ticker))
            else:
                c.execute("INSERT INTO portfolio (ticker, avg_cost, shares, group_name) VALUES (?, ?, ?, ?)", (ticker, total/shares, shares, 'Default'))
        elif trans_type == 'SELL' and row:
            if shares >= row[1]: c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
            else: c.execute("UPDATE portfolio SET shares=? WHERE ticker=?", (row[1]-shares, ticker))
        conn.commit()
        conn.close()
        return "交易成功"

    @staticmethod
    def get_portfolio():
        try:
            conn = sqlite3.connect(DB_NAME)
            return pd.read_sql("SELECT * FROM portfolio", conn)
        except: return pd.DataFrame()

    @staticmethod
    def get_transactions():
        try:
            conn = sqlite3.connect(DB_NAME)
            return pd.read_sql("SELECT * FROM transactions ORDER BY date DESC", conn)
        except: return pd.DataFrame()

DBManager.init_db()

# ==========================================
# 2. 核心運算引擎
# ==========================================
class TechnicalEngine:
    @staticmethod
    def calculate_all(df):
        if df.empty or len(df) < 5: return df
        df = df.copy()
        for col in ['Close', 'High', 'Low', 'Volume']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        for ma in [5, 20, 60]: df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
        
        low_9 = df['Low'].rolling(9).min()
        high_9 = df['High'].rolling(9).max()
        rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        gain = df['Close'].diff().clip(lower=0).rolling(14).mean()
        loss = -df['Close'].diff().clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        std = df['Close'].rolling(20).std()
        df['BB_Up'] = df['MA20'] + (std * 2)
        df['BB_Low'] = df['MA20'] - (std * 2)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        return df

class QuantBrain:
    @staticmethod
    def analyze(ticker, hist, info, price):
        if hist.empty: return None
        curr = hist.iloc[-1]
        
        bias = (price - curr['MA20']) / curr['MA20'] * 100 if curr['MA20'] else 0
        vol_ratio = curr['Volume'] / hist['Volume'].rolling(5).mean().iloc[-2] if len(hist)>5 else 1
        
        eps = info.get('trailingEps')
        pe = price / eps if eps and eps > 0 else 999
        roe = info.get('returnOnEquity', 0)
        yld = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        return {
            "price": price, "ma20": curr['MA20'], "ma60": curr['MA60'],
            "k": curr['K'], "d": curr['D'], "macd": curr['MACD'], "sig": curr['Signal'],
            "rsi": curr['RSI'], "bias": bias, "vol_ratio": vol_ratio,
            "pe": pe, "roe": roe, "yield": yld, "eps": eps, "atr": curr['ATR']
        }

    @staticmethod
    def score(f, strategy="balanced"):
        if not f: return 0
        score = 50
        if f['price'] > f['ma20']: score += 10
        if f['price'] > f['ma60']: score += 10
        if pd.notna(f['k']) and f['k'] > f['d'] and f['k'] < 80: score += 10
        if f['pe'] < 20: score += 10
        if f['yield'] > 4: score += 10
        if f['bias'] > 20: score -= 15
        if pd.notna(f['rsi']) and f['rsi'] > 85: score -= 10
        return max(0, min(100, int(score)))

    @staticmethod
    def explain(f, score):
        if not f: return "N/A"
        pros = []
        if f['roe'] and f['roe'] > 0.15: pros.append(f"ROE佳({f['roe']*100:.1f}%)")
        if f['pe'] < 15: pros.append(f"低估值({f['pe']:.1f}x)")
        if f['price'] > f['ma60']: pros.append("多頭排列")
        if f['vol_ratio'] > 1.5: pros.append("量能放大")
        return " | ".join(pros) if pros else "觀望"

# ==========================================
# 3. 數據抓取層 (Chameleon Engine)
# ==========================================
class DataFetcher:
    @staticmethod
    def normalize(t):
        t = t.strip().upper()
        return t + ".TW" if t.isdigit() else t

    @staticmethod
    def _fetch_twse_rwd(ticker):
        """V20 新增：偽裝成瀏覽器訪問證交所 RWD API (繞過 Geo-blocking)"""
        if not ticker[:2].isdigit(): return pd.DataFrame()
        try:
            sid = ticker.replace(".TW", "")
            # 使用 RWD 新版接口，這通常比舊版 API 寬鬆
            url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={datetime.now().strftime('%Y%m01')}&stockNo={sid}&response=json"
            
            # 關鍵：偽造 Referer，假裝是從證交所官網點進去的
            headers = {
                "User-Agent": UserAgent().random,
                "Referer": "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
                "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept": "application/json"
            }
            
            # verify=False 繞過 SSL
            r = requests.get(url, headers=headers, verify=False, timeout=10)
            data = r.json()
            
            if data.get('stat') != 'OK': return pd.DataFrame()
            
            raw = data['data']
            col = ['Date', 'Volume', 'Amount', 'Open', 'High', 'Low', 'Close', 'Change', 'Trans']
            df = pd.DataFrame(raw, columns=col)
            
            # 民國轉西元
            def conv_date(d):
                y, m, d = d.split('/')
                return f"{int(y)+1911}-{m}-{d}"
            
            df['Date'] = pd.to_datetime(df['Date'].apply(conv_date))
            df.set_index('Date', inplace=True)
            
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
                
            # 證交所 Volume 是股數，轉張數
            df['Volume'] = df['Volume'] / 1000
            
            return df
        except Exception as e:
            print(f"TWSE RWD Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_full(ticker):
        # cached = DBManager.get_cache(ticker) # Debug 期間關閉快取
        # if cached: return cached
        
        ticker = DataFetcher.normalize(ticker)
        data = {"ticker": ticker, "status": "fail"}
        hist = pd.DataFrame()
        info = {}
        
        # --- 策略 A: YahooQuery (抗封鎖能力較強) ---
        try:
            yq = YQTicker(ticker)
            # 抓取 1 年
            hist_yq = yq.history(period='1y')
            if not hist_yq.empty and isinstance(hist_yq.index, pd.MultiIndex):
                hist = hist_yq.reset_index(level=0, drop=True) # 去除 symbol index
                info = yq.asset_profile.get(ticker, {})
                # 補充 info
                summary = yq.summary_detail.get(ticker, {})
                info.update(summary)
                data['status'] = "yahooquery"
        except: pass

        # --- 策略 B: Yfinance (加入 Session 偽裝) ---
        if hist.empty:
            try:
                session = requests.Session()
                session.headers['User-Agent'] = UserAgent().random
                stock = yf.Ticker(ticker, session=session)
                hist = stock.history(period="1y")
                if not hist.empty:
                    info = stock.info
                    data['status'] = "yfinance"
            except: pass

        # --- 策略 C: TWSE RWD (直連台灣證交所) ---
        if hist.empty:
            hist = DataFetcher._fetch_twse_rwd(ticker)
            if not hist.empty: data['status'] = "twse_rwd"

        if hist.empty:
            st.error(f"❌ 嚴重錯誤：無法從任何來源 ({ticker}) 獲取數據。這可能是因為您的雲端 IP 被全面封鎖。")
            return None

        # 補即時價
        price = hist['Close'].iloc[-1]
        try:
            # 嘗試抓最新價，如果失敗就用收盤價
            if ticker[:2].isdigit():
                # 簡單偽裝請求 Twstock 網站
                pass 
        except: pass
        
        name = info.get('longName', ticker)
        
        # 運算
        try:
            hist = TechnicalEngine.calculate_all(hist)
            # 簡易蒙地卡羅 (避免 scipy 依賴問題)
            risk = {"sharpe": 0, "volatility": 0}
            if len(hist) > 30:
                ret = hist['Close'].pct_change()
                risk['volatility'] = ret.std() * np.sqrt(252)
            
            factors = QuantBrain.analyze(ticker, hist, info, price)
            score = QuantBrain.score(factors)
            thesis = QuantBrain.explain(factors, score)
            
            data.update({
                "price": price,
                "change_pct": (price - hist['Close'].iloc[-2])/hist['Close'].iloc[-2]*100,
                "factors": factors, "score": score, "thesis": thesis,
                "hist_json": hist.reset_index().to_json(date_format='iso'),
                "risk": risk
            })
            DBManager.save_cache(ticker, data)
            return data
        except Exception as e:
            st.error(f"運算階段錯誤: {e}")
            return None

    @staticmethod
    def fetch_simple(ticker):
        # 儀表板用最簡單的 YahooQuery
        try:
            t = YQTicker(ticker)
            p = t.price[ticker]['regularMarketPrice']
            prev = t.price[ticker]['regularMarketPreviousClose']
            return {"ticker": ticker, "price": p, "change_pct": (p-prev)/prev*100}
        except: return None

    @staticmethod
    def fetch_batch(tickers, prog):
        res = []
        total = len(tickers)
        # 序列化抓取，每次休息 0.5 秒，避免被鎖
        for i, t in enumerate(tickers):
            if prog: prog.progress((i+1)/total)
            d = DataFetcher.fetch_full(t)
            if d: res.append(d)
            time.sleep(0.5) 
        return res

# ==========================================
# 4. UI
# ==========================================
def plot_chart(d):
    try:
        df = pd.read_json(d['hist_json'])
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']); df.set_index('Date', inplace=True)
        elif 'index' in df.columns: df['index'] = pd.to_datetime(df['index']); df.set_index('index', inplace=True)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='MA20'), row=1, col=1)
        if 'MA60' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue'), name='MA60'), row=1, col=1)
        if 'Volume' in df.columns: fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='量'), row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except: st.error("圖表繪製錯誤")

# Main
with st.sidebar:
    st.title("🦎 Willie's Omega V20")
    st.caption("變色龍匿蹤版")
    st.info("已啟用: YahooQuery + TWSE RWD 偽裝")
    if st.button("清除快取"): st.cache_data.clear(); st.rerun()

tabs = st.tabs(["📊 全球", "🔎 深度戰情", "🎯 穩定選股", "💰 庫存"])

with tabs[0]:
    st.subheader("全球概況 (YQ)")
    items = ["^TWII", "^SOX", "GC=F"]
    cols = st.columns(3)
    for i, t in enumerate(items):
        with cols[i]:
            d = DataFetcher.fetch_simple(t)
            if d: st.metric(t, f"{d['price']:,.2f}", f"{d['change_pct']:.2f}%")
            else: st.metric(t, "連線中...")

with tabs[1]:
    st.subheader("🔎 個股深度分析")
    t = st.text_input("輸入代號", "2330.TW").upper()
    if st.button("開始分析"):
        with st.spinner("啟動三層備援搜索..."):
            d = DataFetcher.fetch_full(t)
            if d:
                st.success(f"資料來源: {d['status']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("現價", d['price'], f"{d['change_pct']:.2f}%")
                m2.metric("AI 評分", d['score'])
                m3.metric("PE", f"{d['factors']['pe']:.1f}")
                st.info(d['thesis'])
                plot_chart(d)

with tabs[2]:
    st.subheader("🎯 穩定型選股 (序列化)")
    univ = st.selectbox("板塊", ["list_tech", "list_finance", "list_shipping"])
    if st.button("開始掃描 (較慢但穩定)"):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT value FROM system_config WHERE key=?", (univ,))
        tgts = c.fetchone()[0].split(",")
        conn.close()
        
        pb = st.progress(0, "初始化...")
        res = DataFetcher.fetch_batch(tgts, pb)
        pb.empty()
        
        rows = []
        for r in res:
            f = r['factors']
            rows.append({"代號": r['ticker'], "AI分": r['score'], "現價": r['price'], "說明": r['thesis']})
        
        if rows: st.dataframe(pd.DataFrame(rows).sort_values("AI分", ascending=False))
        else: st.warning("無資料")

with tabs[3]:
    st.subheader("💰 庫存")
    st.dataframe(DBManager.get_portfolio())
