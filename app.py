import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import time
import json
from datetime import datetime, timedelta

# --- 設定網頁配置 ---
st.set_page_config(page_title="willie 旗艦戰情室 V8", layout="wide", page_icon="🚀")

# ==========================================
# 1. 資料庫層 (SQLite) - 核心升級
# ==========================================
DB_NAME = "joymax_invest.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 建立快取表 (解決 N/A 與速度問題)
    c.execute('''CREATE TABLE IF NOT EXISTS stock_cache
                 (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
    # 建立庫存表 (Portfolio)
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                 (ticker TEXT PRIMARY KEY, cost REAL, shares INTEGER)''')
    conn.commit()
    conn.close()

def get_cached_stock(ticker, ttl_minutes=60):
    """嘗試從資料庫讀取快取，TTL 為過期時間(預設60分鐘)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT data, updated_at FROM stock_cache WHERE ticker=?", (ticker,))
    row = c.fetchone()
    conn.close()
    
    if row:
        data_str, updated_at_str = row
        updated_at = datetime.fromisoformat(updated_at_str)
        if datetime.now() - updated_at < timedelta(minutes=ttl_minutes):
            return json.loads(data_str) # 快取有效
    return None # 快取無效或不存在

def save_to_cache(ticker, data_dict):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    data_str = json.dumps(data_dict)
    c.execute("REPLACE INTO stock_cache (ticker, data, updated_at) VALUES (?, ?, ?)", 
              (ticker, data_str, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def add_portfolio(ticker, cost, shares):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("REPLACE INTO portfolio (ticker, cost, shares) VALUES (?, ?, ?)", (ticker, cost, shares))
    conn.commit()
    conn.close()

def get_portfolio():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM portfolio", conn)
    conn.close()
    return df

def delete_portfolio(ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    conn.commit()
    conn.close()

# 初始化資料庫
init_db()

# ==========================================
# 2. 技術指標計算引擎
# ==========================================
def calculate_ta(df):
    # KD 指標
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ==========================================
# 3. 核心數據抓取 (整合快取)
# ==========================================
def fetch_stock_data(ticker, use_cache=True):
    ticker = ticker.strip().upper()
    if not ticker.endswith(".TW") and not ticker.endswith(".TWO"): ticker += ".TW"
    
    # 1. 嘗試讀快取
    if use_cache:
        cached = get_cached_stock(ticker)
        if cached: return cached

    # 2. 沒快取則抓取
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty: return None
        
        # 計算技術指標
        hist = calculate_ta(hist)
        
        # 整理基本資料
        info = stock.info
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        
        # 萃取需要儲存的數據
        data = {
            "price": current,
            "change_pct": (current - prev) / prev * 100,
            "volume": hist['Volume'].iloc[-1],
            "eps": info.get('trailingEps') or info.get('forwardEps'),
            "pe": current / (info.get('trailingEps') or 1) if info.get('trailingEps') else None,
            "yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "ma20": hist['Close'].rolling(20).mean().iloc[-1],
            "ma60": hist['Close'].rolling(60).mean().iloc[-1],
            "k": hist['K'].iloc[-1],
            "d": hist['D'].iloc[-1],
            "macd": hist['MACD'].iloc[-1],
            "macd_sig": hist['Signal'].iloc[-1],
            "rsi": hist['RSI'].iloc[-1],
            "history_close": hist['Close'].to_json(), # 存圖表用
            "name": info.get('longName', ticker)
        }
        
        # 寫入快取
        save_to_cache(ticker, data)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

# ==========================================
# 4. AI 解盤引擎 (V8 增強版)
# ==========================================
def generate_ai_report(ticker, d):
    date_str = datetime.now().strftime("%Y/%m/%d")
    
    # 技術訊號判讀
    ta_signal = []
    if d['k'] > d['d']: ta_signal.append("KD黃金交叉(偏多)")
    else: ta_signal.append("KD死亡交叉(偏空)")
    
    if d['macd'] > d['macd_sig']: ta_signal.append("MACD柱狀體翻紅")
    
    if d['rsi'] > 70: ta_signal.append("RSI過熱(恐拉回)")
    elif d['rsi'] < 30: ta_signal.append("RSI超賣(醞釀反彈)")
    
    ta_str = "、".join(ta_signal)

    # 殖利率判讀
    yield_str = f"預估殖利率 {d['yield']:.2f}%" if d['yield'] > 0 else "無配息資訊"
    
    full_text = f"""
【Joymax 智能投顧】{d['name']} ({ticker})
📅 日期：{date_str}
------------------------
💰 收盤：{d['price']:.1f} ({d['change_pct']:+.2f}%)
📊 殖利率：{yield_str}

🤖 AI 多維度解析：
1. 籌碼/型態：{ta_str}。
2. 均線趨勢：股價{"站上" if d['price'] > d['ma20'] else "跌破"}月線，{"站上" if d['price'] > d['ma60'] else "跌破"}季線。
3. 估值評價：本益比 {d['pe']:.1f} 倍 ({ "偏低" if d['pe'] and d['pe']<12 else "合理" if d['pe'] and d['pe']<20 else "偏高" })。

💡 綜合建議：
技術面出現 {ta_signal[0]} 訊號，配合 {yield_str} 防護，建議{"分批佈局" if d['change_pct']>0 else "觀察支撐"}。
    """
    return full_text.strip()

# ==========================================
# 5. UI 介面
# ==========================================

# --- 側邊欄導航 ---
with st.sidebar:
    st.title("Joymax V8 旗艦版")
    page = st.radio("前往頁面", ["📊 戰情儀表板", "💰 我的庫存管理", "🚀 戰術掃描"])
    st.markdown("---")
    
    if page == "💰 我的庫存管理":
        st.subheader("新增庫存")
        p_ticker = st.text_input("代號", "2330").upper()
        p_cost = st.number_input("平均成本", min_value=0.0, value=1000.0)
        p_shares = st.number_input("股數 (張數*1000)", min_value=1, value=1000)
        if st.button("💾 儲存/更新持股"):
            if not p_ticker.endswith("TW"): p_ticker += ".TW"
            add_portfolio(p_ticker, p_cost, p_shares)
            st.success(f"已儲存 {p_ticker}")
            time.sleep(1)
            st.rerun()

# --- 頁面 1: 戰情儀表板 (含個股詳細分析) ---
if page == "📊 戰情儀表板":
    st.title("📊 市場總覽與個股分析")
    
    # 大盤指數
    cols = st.columns(4)
    indices = {"^TWII": "加權指數", "^TWOII": "櫃買指數", "^SOX": "費半指數", "^IXIC": "那斯達克"}
    for i, (k, v) in enumerate(indices.items()):
        d = fetch_stock_data(k) # 指數也有快取了！
        with cols[i]:
            if d: st.metric(v, f"{d['price']:,.0f}", f"{d['change_pct']:.2f}%")
            else: st.metric(v, "Loading...")
    
    st.divider()
    
    # 個股查詢 (整合所有功能)
    col_input, col_btn = st.columns([3, 1])
    ticker = col_input.text_input("輸入個股代號 (支援快取秒開)", "2330.TW").upper()
    if col_btn.button("🔍 深度分析"):
        d = fetch_stock_data(ticker, use_cache=False) # 強制更新
    else:
        d = fetch_stock_data(ticker) # 預設讀快取

    if d:
        st.subheader(f"📌 {d['name']} ({ticker})")
        
        # 1. 核心指標
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{d['price']}", f"{d['change_pct']:.2f}%")
        c2.metric("KD (K/D)", f"{d['k']:.1f} / {d['d']:.1f}")
        c3.metric("RSI (強弱)", f"{d['rsi']:.1f}")
        c4.metric("殖利率", f"{d['yield']:.2f}%")
        
        # 2. AI 報告
        with st.expander("🤖 點擊查看 AI 智能投顧報告 (含複製功能)", expanded=True):
            report = generate_ai_report(ticker, d)
            st.code(report, language="text")
            
        # 3. 圖表 (K線與技術指標)
        # 這裡還原歷史股價
        hist_series = pd.read_json(d['history_close'], typ='series')
        st.line_chart(hist_series)
        
        # 4. 新聞傳送門
        st.markdown("📰 **相關新聞**")
        sid = ticker.replace(".TW", "").replace(".TWO", "")
        st.link_button("Yahoo 新聞", f"https://tw.stock.yahoo.com/quote/{sid}")

# --- 頁面 2: 庫存管理 (新功能！) ---
elif page == "💰 我的庫存管理":
    st.title("💰 資產管理中心")
    
    df_port = get_portfolio()
    
    if df_port.empty:
        st.info("目前沒有庫存，請從左側側邊欄新增。")
    else:
        # 計算即時損益
        total_market_val = 0
        total_cost_val = 0
        
        portfolio_data = []
        bar = st.progress(0, "計算庫存現值中...")
        
        for i, row in df_port.iterrows():
            bar.progress((i+1)/len(df_port))
            d = fetch_stock_data(row['ticker']) # 讀快取，速度快
            current_price = d['price'] if d else row['cost'] # 抓不到就用成本價暫代
            
            market_val = current_price * row['shares']
            cost_val = row['cost'] * row['shares']
            pnl = market_val - cost_val
            pnl_pct = (pnl / cost_val) * 100
            
            total_market_val += market_val
            total_cost_val += cost_val
            
            portfolio_data.append({
                "代號": row['ticker'],
                "股數": row['shares'],
                "成本": row['cost'],
                "現價": current_price,
                "市值": int(market_val),
                "損益 $": int(pnl),
                "報酬率 %": round(pnl_pct, 2)
            })
            
        bar.empty()
        
        # 總結
        total_pnl = total_market_val - total_cost_val
        total_pnl_pct = (total_pnl / total_cost_val * 100) if total_cost_val > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("總市值", f"${total_market_val:,.0f}")
        c2.metric("總損益", f"${total_pnl:,.0f}", f"{total_pnl_pct:.2f}%")
        c3.metric("持股檔數", f"{len(portfolio_data)}")
        
        # 詳細表格
        st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)
        
        # 刪除功能
        del_ticker = st.selectbox("選擇要刪除的持股", df_port['ticker'])
        if st.button("🗑️ 刪除選定持股"):
            delete_portfolio(del_ticker)
            st.rerun()
            
        # 資產配置圖
        fig = px.pie(portfolio_data, values='市值', names='代號', title='資產配置分布')
        st.plotly_chart(fig)

# --- 頁面 3: 戰術掃描 (保留 V6 功能但加上快取加速) ---
elif page == "🚀 戰術掃描":
    st.title("🚀 市場雷達")
    
    source = st.radio("掃描範圍", ["Top 20", "自訂清單"])
    tickers = ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2881.TW", "0050.TW"] # 預設簡化
    
    if source == "自訂清單":
        user_list = st.text_area("輸入代號", "2330, 2603")
        tickers = [x.strip() for x in user_list.replace("\n", ",").split(",") if x]
    
    if st.button("開始掃描"):
        data_list = []
        bar = st.progress(0)
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            # 使用快取抓取，速度會越來越快
            d = fetch_stock_data(t)
            if d:
                data_list.append({
                    "代號": t, "現價": d['price'], "漲跌%": f"{d['change_pct']:.2f}",
                    "KD": f"{d['k']:.0f}/{d['d']:.0f}", "RSI": f"{d['rsi']:.0f}",
                    "殖利率": f"{d['yield']:.1f}%", "PE": f"{d['pe']:.1f}" if d['pe'] else "N/A"
                })
        bar.empty()
        st.dataframe(pd.DataFrame(data_list), use_container_width=True)
