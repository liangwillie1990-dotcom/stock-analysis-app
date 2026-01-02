import streamlit as st
import yfinance as yf
import twstock
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
import time
import json
from datetime import datetime, timedelta

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 戰情室 V9.1", layout="wide", page_icon="🚀")

# ==========================================
# 1. 資料庫層
# ==========================================
DB_NAME = "joymax_invest.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS stock_cache
                     (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                     (ticker TEXT PRIMARY KEY, cost REAL, shares INTEGER)''')
        conn.commit()
        conn.close()
    except: pass

def get_cached_stock(ticker, ttl_minutes=30):
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

def save_to_cache(ticker, data_dict):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("REPLACE INTO stock_cache (ticker, data, updated_at) VALUES (?, ?, ?)", 
                  (ticker, json.dumps(data_dict), datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except: pass

def get_portfolio():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM portfolio", conn)
        conn.close()
        return df
    except: return pd.DataFrame()

def add_portfolio(ticker, cost, shares):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
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
# 2. 混合式抓取引擎 (修正 lxml 依賴)
# ==========================================
def fetch_stock_data(ticker, use_cache=True):
    ticker = ticker.strip().upper()
    is_tw_stock = ticker[:2].isdigit()
    
    if is_tw_stock and not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        yahoo_ticker = ticker + ".TW"
    else:
        yahoo_ticker = ticker

    if use_cache:
        cached = get_cached_stock(yahoo_ticker)
        if cached: return cached

    data = {}
    
    # === 引擎 A: Twstock ===
    if is_tw_stock:
        try:
            stock_id = ticker.replace(".TW", "").replace(".TWO", "")
            real = twstock.realtime.get(stock_id)
            if real['success']:
                current_price = float(real['realtime']['latest_trade_price'])
                data['price'] = current_price
                data['name'] = real['info']['name']
        except:
            pass 

    # === 引擎 B: Yahoo Finance ===
    try:
        stock = yf.Ticker(yahoo_ticker)
        hist = stock.history(period="6mo")
        
        if hist.empty: return None

        if 'price' not in data:
            data['price'] = hist['Close'].iloc[-1]
            
        close = hist['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        low_min = hist['Low'].rolling(9).min()
        high_max = hist['High'].rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()

        prev = hist['Close'].iloc[-2]
        change_pct = (data['price'] - prev) / prev * 100

        pe = None
        yield_val = 0
        eps = None
        try:
            info = stock.info
            eps = info.get('trailingEps')
            if eps: pe = data['price'] / eps
            yield_val = info.get('dividendYield', 0) * 100
            if 'name' not in data: data['name'] = info.get('longName', ticker)
        except: pass

        data.update({
            "change_pct": change_pct,
            "volume": hist['Volume'].iloc[-1],
            "pe": pe,
            "eps": eps,
            "yield": yield_val,
            "k": k.iloc[-1],
            "d": d.iloc[-1],
            "rsi": rsi.iloc[-1],
            "ma20": close.rolling(20).mean().iloc[-1],
            "history_close": hist['Close'].to_json()
        })
        
        save_to_cache(yahoo_ticker, data)
        return data

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# ==========================================
# 3. AI 報告
# ==========================================
def generate_ai_report(ticker, d):
    ta = []
    if d['k'] > d['d']: ta.append("KD金叉")
    else: ta.append("KD死叉")
    if d['rsi'] > 80: ta.append("RSI過熱")
    elif d['rsi'] < 20: ta.append("RSI超賣")
    
    return f"""
【Joymax 混合引擎報告】{d.get('name', ticker)}
💰 收盤：{d['price']:.1f} ({d['change_pct']:+.2f}%)
📊 訊號：{', '.join(ta)}
🤖 建議：股價{"站上" if d['price']>d['ma20'] else "跌破"}月線，{"多頭格局" if d['change_pct']>0 else "短線偏弱"}。
    """.strip()

# ==========================================
# 4. UI 介面
# ==========================================
with st.sidebar:
    st.title("Joymax V9.1 混合引擎")
    page = st.radio("功能選單", ["📊 戰情儀表板", "💰 庫存管理", "🚀 戰術掃描"])
    st.info("💡 V9.1：修正大盤迴圈錯誤，補上 lxml 依賴。")

    if page == "💰 庫存管理":
        st.subheader("新增庫存")
        t = st.text_input("代號", "2330")
        c = st.number_input("成本", value=1000.0)
        s = st.number_input("股數", value=1000)
        if st.button("儲存"):
            add_portfolio(t, c, s)
            st.rerun()

if page == "📊 戰情儀表板":
    st.title("📊 市場總覽")
    
    cols = st.columns(4)
    indices = {"^TWII": "加權指數", "^TWOII": "櫃買指數", "^SOX": "費半指數", "^IXIC": "那斯達克"}
    
    # 這裡修復了 V9 的迴圈錯誤
    for i, (k, v) in enumerate(indices.items()):
        with cols[i]:
            d = fetch_stock_data(k)
            if d: st.metric(v, f"{d['price']:,.0f}", f"{d['change_pct']:.2f}%")
            else: st.metric(v, "N/A")

    st.divider()
    
    ticker = st.text_input("輸入個股代號", "2330").upper()
    if st.button("深度分析"):
        d = fetch_stock_data(ticker, use_cache=False)
    else:
        d = fetch_stock_data(ticker)

    if d:
        st.subheader(f"📌 {d.get('name', ticker)}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{d['price']}", f"{d['change_pct']:.2f}%")
        c2.metric("KD", f"{d['k']:.0f}/{d['d']:.0f}")
        c3.metric("RSI", f"{d['rsi']:.1f}")
        c4.metric("殖利率", f"{d['yield']:.2f}%")
        
        st.code(generate_ai_report(ticker, d))
        st.line_chart(pd.read_json(d['history_close'], typ='series'))

elif page == "💰 庫存管理":
    st.title("💰 我的庫存")
    df = get_portfolio()
    if not df.empty:
        res = []
        bar = st.progress(0, "更新股價中...")
        for i, row in df.iterrows():
            bar.progress((i+1)/len(df))
            d = fetch_stock_data(row['ticker'])
            curr = d['price'] if d else row['cost']
            res.append({
                "代號": row['ticker'], "現價": curr, 
                "損益": int((curr - row['cost']) * row['shares']),
                "報酬率%": round((curr - row['cost'])/row['cost']*100, 2)
            })
        bar.empty()
        st.dataframe(pd.DataFrame(res), use_container_width=True)
    else:
        st.info("無庫存資料")

elif page == "🚀 戰術掃描":
    st.title("🚀 快速掃描")
    default = "2330, 2317, 2603, 3231, 0050"
    user = st.text_area("代號列表", default)
    if st.button("掃描"):
        ts = [x.strip() for x in user.split(",")]
        res = []
        bar = st.progress(0)
        for i, t in enumerate(ts):
            bar.progress((i+1)/len(ts))
            d = fetch_stock_data(t)
            if d:
                res.append({"代號": t, "現價": d['price'], "漲跌%": f"{d['change_pct']:.2f}"})
        bar.empty()
        st.dataframe(pd.DataFrame(res), use_container_width=True)
