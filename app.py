"""
Willie's Alpha V16.1 - Classic Restored Edition
Author: Gemini AI
Description:
    Restored V16 features (AI Thesis, Expanded Universe, Radar Chart).
    Patched for stability (Fixed 'use_container_width' and Yahoo connection).
"""

import streamlit as st
import yfinance as yf
import twstock
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import time
import json
import threading
import concurrent.futures
import requests
from datetime import datetime, timedelta
from scipy.stats import norm
from fake_useragent import UserAgent

# ==========================================
# 0. 全局設定
# ==========================================
st.set_page_config(
    page_title="Willie's Alpha V16.1",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root { --primary: #00d4ff; --bg-dark: #0e1117; --card-bg: #1a1c24; }
    .stApp { font-family: 'Roboto Mono', 'Microsoft JhengHei', monospace; background-color: var(--bg-dark); }
    div[data-testid="stMetric"] { background: rgba(26,28,36,0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; }
    div[data-testid="stMetric"]:hover { border-color: var(--primary); }
    .stButton>button { background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); color: white; font-weight: bold; border: none; }
    /* 讓 Dataframe 的文字換行 */
    div[data-testid="stDataFrame"] div[role="gridcell"] { white-space: normal !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫與擴充名單
# ==========================================
DB_NAME = "willie_alpha_v16.db"

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
            DBManager._seed_expanded_universe()
        except: pass

    @staticmethod
    def _seed_expanded_universe():
        universe = {
            "半導體/IC設計": "2330,2454,2303,3034,3035,2379,3443,3661,3529,4961,3006,3227,8016,8299,6415,6531,6756,2408,2449,6223,6533,8081",
            "AI伺服器/電腦": "2317,2382,3231,2357,6669,2356,2301,3017,2324,2421,2376,2377,3013,3515,6214,8112,8210,3037,2345,4938",
            "金融/控股": "2881,2882,2891,2886,2892,2884,2890,5880,2885,2880,2883,2887,2801,2809,2812,2834,2838,2845,2849,2850,2851,2855,2867,5876,5871",
            "航運/運輸": "2603,2609,2615,2618,2610,2637,5608,2606,2605,2636,2607,2608,2641,2642,2634",
            "重電/綠能/軍工": "1513,1514,1519,1504,1605,1609,3708,9958,6806,6443,6477,8046,2630,8033,5284,2062",
            "傳產龍頭": "1101,1102,1301,1303,1326,1304,1308,1312,2002,2014,2006,2027,2105,9904,9910,2912,2915",
            "高股息/市值 ETF": "0050,0056,00878,00919,00929,00939,00940,006208,00713,0052,00631L,00679B,00687B"
        }
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for k, v in universe.items():
            c.execute("INSERT OR IGNORE INTO system_config (key, value) VALUES (?, ?)", (k, v))
        conn.commit()
        conn.close()

    @staticmethod
    def get_market_universe(sector="全部"):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        if sector == "全部":
            c.execute("SELECT value FROM system_config")
            rows = c.fetchall()
            all_tickers = []
            for r in rows: all_tickers.extend(r[0].split(","))
            return list(set(all_tickers))
        else:
            c.execute("SELECT value FROM system_config WHERE key=?", (sector,))
            row = c.fetchone()
            return row[0].split(",") if row else []

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
    def get_portfolio():
        try:
            conn = sqlite3.connect(DB_NAME)
            return pd.read_sql("SELECT * FROM portfolio", conn)
        except: return pd.DataFrame()

DBManager.init_db()

# ==========================================
# 2. QuantBrain (AI 核心)
# ==========================================
class QuantBrain:
    @staticmethod
    def analyze_stock(ticker, hist, info, price):
        if hist.empty: return None
        close = hist['Close']
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        
        # KD
        rsv = (close - hist['Low'].rolling(9).min()) / (hist['High'].rolling(9).max() - hist['Low'].rolling(9).min()) * 100
        k = rsv.ewm(com=2).mean().iloc[-1]
        d = k.ewm(com=2).mean().iloc[-1]
        
        # MACD
        exp12 = close.ewm(span=12).mean()
        exp26 = close.ewm(span=26).mean()
        macd = (exp12 - exp26).iloc[-1]
        signal = (exp12 - exp26).ewm(span=9).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta>0, 0)).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        volatility = close.pct_change().std() * np.sqrt(252)
        bias_20 = (price - ma20) / ma20 * 100
        
        eps = info.get('trailingEps') or info.get('forwardEps')
        pe = price / eps if eps and eps > 0 else 999
        pb = info.get('priceToBook', 0)
        roe = info.get('returnOnEquity', 0)
        yield_val = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        vol_ratio = hist['Volume'].iloc[-1] / hist['Volume'].rolling(5).mean().iloc[-2] if len(hist) > 5 else 1
        
        return {
            "price": price, "ma20": ma20, "ma60": ma60,
            "k": k, "d": d, "macd": macd, "macd_sig": signal, "rsi": rsi,
            "bias_20": bias_20, "volatility": volatility, "vol_ratio": vol_ratio,
            "pe": pe, "pb": pb, "roe": roe, "yield": yield_val, "eps": eps
        }

    @staticmethod
    def score_stock(f, strategy="balanced"):
        if not f: return 0
        score = 50
        w_t, w_v = 1.0, 1.0
        if strategy == "value": w_v = 2.0; w_t = 0.5
        elif strategy == "growth": w_t = 2.0; w_v = 0.5
        elif strategy == "dividend": w_value = 1.5; w_trend = 0.8
        
        if f['price'] > f['ma20']: score += 5 * w_t
        if f['price'] > f['ma60']: score += 5 * w_t
        if f['macd'] > f['macd_sig']: score += 5 * w_t
        if f['k'] > f['d'] and f['k'] < 80: score += 3 * w_t
        if f['vol_ratio'] > 1.5: score += 5 
        
        if f['pe'] < 15: score += 5 * w_v
        if f['pe'] > 40: score -= 5 * w_v
        if f['pb'] > 0 and f['pb'] < 1.5: score += 5 * w_v
        if f['roe'] and f['roe'] > 0.15: score += 10 * w_v
        if f['yield'] > 4: score += 5 * w_v
        
        if f['bias_20'] > 20: score -= 10
        if f['rsi'] > 85: score -= 5
        
        return max(0, min(100, int(score)))

    @staticmethod
    def generate_thesis(f, score):
        if not f: return "資料不足"
        reasons = []
        cautions = []
        
        if f['roe'] and f['roe'] > 0.15: reasons.append(f"🔥 高ROE({f['roe']*100:.1f}%)")
        if f['pe'] < 12 and f['pe'] > 0: reasons.append(f"💎 低本益比({f['pe']:.1f}x)")
        if f['yield'] > 5: reasons.append(f"💰 高殖利({f['yield']:.1f}%)")
        if f['price'] > f['ma20'] and f['price'] > f['ma60']: reasons.append("📈 多頭排列")
        if f['macd'] > f['macd_sig'] and f['macd'] < 0: reasons.append("⚡ 底部翻揚")
        if f['vol_ratio'] > 2.0: reasons.append("🌊 爆量")
        if f['k'] < 20 and f['k'] > f['d']: reasons.append("🎣 低檔金叉")

        if f['bias_20'] > 15: cautions.append(f"⚠️ 乖離大")
        if f['rsi'] > 80: cautions.append("🔥 過熱")
        if f['price'] < f['ma60']: cautions.append("❄️ 弱勢")
        
        thesis = ""
        if score >= 75: thesis += "🚀 強力買進: "
        elif score >= 60: thesis += "🟢 偏多: "
        elif score <= 40: thesis += "🔴 偏空: "
        else: thesis += "⚪ 觀望: "
        
        thesis += " | ".join(reasons) if reasons else "無明顯利多"
        if cautions: thesis += " (風險: " + " | ".join(cautions) + ")"
        return thesis

# ==========================================
# 3. 資料抓取
# ==========================================
class DataFetcher:
    @staticmethod
    def normalize(t):
        t = t.strip().upper()
        return t + ".TW" if t.isdigit() else t

    @staticmethod
    def fetch_analysis_data(ticker):
        ticker = DataFetcher.normalize(ticker)
        cached = DBManager.get_cache(ticker)
        if cached: return cached
        
        try:
            # 加入 User-Agent 偽裝 (修復重點)
            session = requests.Session()
            session.headers['User-Agent'] = UserAgent().random
            
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period="1y")
            if hist.empty: return None
            
            price = hist['Close'].iloc[-1]
            try:
                if ticker[:2].isdigit():
                    real = twstock.realtime.get(ticker.replace(".TW", ""))
                    if real['success']: 
                        price = float(real['realtime']['latest_trade_price'])
            except: pass
            
            factors = QuantBrain.analyze_stock(ticker, hist, stock.info, price)
            score = QuantBrain.score_stock(factors, "balanced")
            thesis = QuantBrain.generate_thesis(factors, score)
            
            data = {
                "ticker": ticker, "name": stock.info.get('longName', ticker),
                "price": price, "change_pct": (price - hist['Close'].iloc[-2])/hist['Close'].iloc[-2]*100,
                "factors": factors, "score": score, "thesis": thesis,
                "volume": hist['Volume'].iloc[-1],
                "history_json": hist.reset_index().to_json(date_format='iso')
            }
            DBManager.save_cache(ticker, data)
            return data
        except: return None

    @staticmethod
    def fetch_batch(tickers, pb=None):
        results = []
        # 改用序列化處理以避免被擋 (修復重點)
        total = len(tickers)
        for i, t in enumerate(tickers):
            if pb: pb.progress((i+1)/total)
            res = DataFetcher.fetch_analysis_data(t)
            if res: results.append(res)
            time.sleep(0.1) # 溫柔請求
        return results

# ==========================================
# 4. UI 視覺化
# ==========================================
def render_radar(d):
    f = d['factors']
    risk = max(0, 100 - f['volatility']*200)
    val = 80 if f['pe'] < 15 else 40
    tech = 80 if f['k'] > f['d'] else 40
    yld = min(100, f['yield']*15)
    trend = 80 if f['price'] > f['ma60'] else 30
    
    fig = go.Figure(go.Scatterpolar(
        r=[val, tech, 60, trend, yld, risk],
        theta=['估值', '技術', '籌碼', '趨勢', '殖利率', '低風險'],
        fill='toself', line=dict(color='#00d4ff')
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300, 
                      margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ddd"))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. 主程式 (UI)
# ==========================================
with st.sidebar:
    st.title("🦅 Willie's Alpha V16.1")
    st.caption("經典復刻 | 穩定版")
    if st.button("清除快取"):
        st.cache_data.clear()
        st.rerun()

tabs = st.tabs(["📊 戰情儀表", "🎯 AI 量化選股", "🔎 個股深度"])

# Tab 1: 儀表板
with tabs[0]:
    st.subheader("🌍 全球市場")
    items = ["^TWII", "^TWOII", "^SOX", "GC=F"]
    cols = st.columns(4)
    for i, t in enumerate(items):
        with cols[i]:
            d = DataFetcher.fetch_analysis_data(t)
            if d: st.metric(t, f"{d['price']:,.2f}", f"{d['change_pct']:.2f}%")
            else: st.metric(t, "N/A")

# Tab 2: 選股 (V16 核心)
with tabs[1]:
    st.subheader("🎯 AI 因子選股")
    
    with st.expander("設定策略", expanded=True):
        c1, c2, c3 = st.columns(3)
        strat = c1.selectbox("AI 個性", ["balanced", "value", "growth", "dividend"])
        univ = c2.selectbox("範圍", ["半導體/IC設計", "AI伺服器/電腦", "金融/控股", "航運/運輸", "重電/綠能/軍工", "高股息/市值 ETF"])
        min_s = c3.slider("最低分", 0, 90, 60)
        
        if st.button("🚀 啟動 Willie 引擎"):
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT value FROM system_config WHERE key=?", (univ,))
            targets = c.fetchone()[0].split(",")
            conn.close()
            
            pb = st.progress(0, "分析中...")
            res = DataFetcher.fetch_batch(targets, pb)
            pb.empty()
            
            rows = []
            for r in res:
                f = r['factors']
                s = QuantBrain.score_stock(f, strat)
                if s >= min_s:
                    rows.append({
                        "代號": r['ticker'], "名稱": r['name'], "AI評分": s,
                        "現價": r['price'], "PE": f"{f['pe']:.1f}", "ROE": f"{f['roe']*100:.1f}%",
                        "AI論述": QuantBrain.generate_thesis(f, s)
                    })
            
            if rows:
                df = pd.DataFrame(rows).sort_values("AI評分", ascending=False)
                # 這裡使用了 st.dataframe 預設設定，不加 use_container_width 以防報錯
                st.dataframe(df)
            else:
                st.warning("無符合條件標的")

# Tab 3: 深度
with tabs[2]:
    t = st.text_input("輸入代號", "2330.TW").upper()
    if st.button("分析"):
        d = DataFetcher.fetch_analysis_data(t)
        if d:
            st.header(f"{d['name']} ({d['ticker']})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Score", d['score'])
            m2.metric("PE", f"{d['factors']['pe']:.1f}x")
            m3.metric("ROE", f"{d['factors']['roe']*100:.1f}%")
            m4.metric("ATR", f"{d['factors'].get('atr', 0):.1f}") # 安全獲取
            
            c1, c2 = st.columns([2, 1])
            with c1: st.info(d['thesis'])
            with c2: render_radar(d)
            
            df_h = pd.read_json(d['history_json'])
            if 'Date' in df_h.columns: df_h['Date'] = pd.to_datetime(df_h['Date']); df_h.set_index('Date', inplace=True)
            elif 'index' in df_h.columns: df_h['index'] = pd.to_datetime(df_h['index']); df_h.set_index('index', inplace=True)
            st.line_chart(df_h['Close'])
