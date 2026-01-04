"""
Joymax 戰情室 V10.3 (Sector Navigation)
Version: V10.3
Feature: Added pre-defined sector lists for easy scanning without manual input.
"""

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
st.set_page_config(page_title="Joymax 戰情室 V10.3", layout="wide", page_icon="🧭")

# ==========================================
# 0. 內建板塊名單 (V10.3 新增)
# ==========================================
SECTOR_LISTS = {
    "🔹 自訂/預設": "2330, 2317, 2454, 2603, 2881, 2308, 0050",
    "🤖 AI 伺服器": "2317, 2382, 3231, 2357, 6669, 2356, 3017, 2324, 2376, 2421, 3515",
    "💻 半導體/IC設計": "2330, 2454, 2303, 3034, 3035, 2379, 3443, 3661, 4961, 3006, 2408",
    "🏦 金融存股": "2881, 2882, 2891, 2886, 2892, 2884, 2890, 5880, 2885, 2880, 2883, 2887",
    "🚢 航運三雄+散裝": "2603, 2609, 2615, 2618, 2610, 2637, 5608, 2606, 2605",
    "⚡ 重電與綠能": "1513, 1514, 1519, 1504, 1605, 1609, 3708, 9958, 6806",
    "💰 高股息 ETF": "0056, 00878, 00919, 00929, 00940, 00713, 00939, 00918, 0050"
}

# ==========================================
# 1. 資料庫層
# ==========================================
DB_NAME = "joymax_v10.db"

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
# 2. 抓取邏輯 (V10 核心)
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
    
    # Twstock
    if is_tw_stock:
        try:
            stock_id = ticker.replace(".TW", "").replace(".TWO", "")
            real = twstock.realtime.get(stock_id)
            if real['success']:
                data['price'] = float(real['realtime']['latest_trade_price'])
                data['name'] = real['info']['name']
        except: pass

    # Yahoo
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
        eps = None
        yield_val = 0
        try:
            info = stock.info
            eps = info.get('trailingEps') or info.get('forwardEps')
            if eps: pe = data['price'] / eps
            yield_val = info.get('dividendYield', 0) * 100
            if 'name' not in data: data['name'] = info.get('longName', ticker)
        except: pass

        valuation = {}
        if eps and not hist.empty:
            pe_series = hist['Close'] / eps
            pe_min = pe_series.min()
            pe_mean = pe_series.mean()
            pe_max = pe_series.max()
            valuation = {"cheap": eps * pe_min, "fair": eps * pe_mean, "expensive": eps * pe_max}

        data.update({
            "change_pct": change_pct, "volume": hist['Volume'].iloc[-1],
            "pe": pe, "eps": eps, "yield": yield_val,
            "k": k.iloc[-1], "d": d.iloc[-1], "rsi": rsi.iloc[-1],
            "ma20": close.rolling(20).mean().iloc[-1],
            "history_close": hist['Close'].to_json(),
            "valuation": valuation, "high_52": hist['High'].max(), "low_52": hist['Low'].min()
        })
        
        save_to_cache(yahoo_ticker, data)
        return data

    except Exception as e:
        print(f"Error: {e}")
        return None

# ==========================================
# 3. AI 報告
# ==========================================
def generate_ai_report(ticker, d):
    ta = []
    if d['k'] > d['d']: ta.append("KD金叉")
    else: ta.append("KD死叉")
    
    val_str = "N/A"
    pe_str = "N/A"
    if d['pe']:
        pe_str = f"{d['pe']:.1f}"
        if d['pe'] < 15: val_str = "低估"
        elif d['pe'] < 20: val_str = "合理"
        else: val_str = "偏高"

    return f"""
【Joymax 智能日報】{d.get('name', ticker)}
💰 收盤：{d['price']:.1f} ({d['change_pct']:+.2f}%)
📊 技術：{', '.join(ta)} | RSI: {d['rsi']:.1f}
💎 估值：PE {pe_str}倍 ({val_str})
    """.strip()

# ==========================================
# 4. UI 介面
# ==========================================
with st.sidebar:
    st.title("Joymax V10.3")
    st.caption("板塊導航版")
    page = st.radio("功能選單", ["📊 戰情儀表板", "🚀 戰術掃描", "💰 庫存管理"])
    st.success("功能：新增熱門板塊清單")
    
    if page == "💰 庫存管理":
        st.subheader("新增庫存")
        t = st.text_input("代號", "2330")
        c = st.number_input("成本", value=1000.0)
        s = st.number_input("股數", value=1000)
        if st.button("儲存"):
            add_portfolio(t, c, s)
            st.rerun()

if page == "📊 戰情儀表板":
    st.title("📊 市場總覽與個股分析")
    
    cols = st.columns(4)
    indices = {"^TWII": "加權指數", "^TWOII": "櫃買指數", "^SOX": "費半指數", "^IXIC": "那斯達克"}
    for i, (k, v) in enumerate(indices.items()):
        with cols[i]:
            d = fetch_stock_data(k)
            if d: st.metric(v, f"{d['price']:,.0f}", f"{d['change_pct']:.2f}%")
            else: st.metric(v, "N/A")
    st.divider()
    
    col_input, col_btn = st.columns([3, 1])
    ticker = col_input.text_input("輸入個股代號", "2330.TW").upper()
    
    if col_btn.button("🔍 深度分析"):
        d = fetch_stock_data(ticker, use_cache=False)
    else:
        d = fetch_stock_data(ticker)

    if d:
        st.subheader(f"📌 {d.get('name', ticker)}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{d['price']}", f"{d['change_pct']:.2f}%")
        pe_d = f"{d['pe']:.1f}x" if d['pe'] else "N/A"
        c2.metric("本益比", pe_d)
        c3.metric("KD", f"{d['k']:.0f}/{d['d']:.0f}")
        c4.metric("殖利率", f"{d['yield']:.2f}%")

        if d.get('valuation'):
            val = d['valuation']
            st.write("### 💎 本益比目標價")
            v1, v2, v3 = st.columns(3)
            v1.metric("保守", f"{val['cheap']:.1f}")
            v2.metric("合理", f"{val['fair']:.1f}")
            v3.metric("昂貴", f"{val['expensive']:.1f}")
            
            fig = go.Figure()
            curr = d['price']
            fig.add_trace(go.Scatter(x=[curr], y=[0], mode='markers+text', text=["現價"], marker=dict(size=15, color='black')))
            fig.add_trace(go.Bar(x=[val['cheap']], y=[0], orientation='h', name='便宜', marker_color='green', opacity=0.3))
            fig.add_trace(go.Bar(x=[val['fair']-val['cheap']], y=[0], base=val['cheap'], orientation='h', name='合理', marker_color='blue', opacity=0.3))
            fig.add_trace(go.Bar(x=[val['expensive']-val['fair']], y=[0], base=val['fair'], orientation='h', name='昂貴', marker_color='red', opacity=0.3))
            fig.update_layout(height=150, barmode='stack', yaxis=dict(showticklabels=False), margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.code(generate_ai_report(ticker, d))
        st.line_chart(pd.read_json(d['history_close'], typ='series'))

# --- 頁面 2: 戰術掃描 (V10.3 優化重點) ---
elif page == "🚀 戰術掃描":
    st.title("🚀 市場雷達")
    
    # --- 新增：板塊選擇器 ---
    st.info("💡 提示：使用下方選單快速載入熱門股票，無需手動輸入。")
    selected_sector = st.selectbox("📂 選擇觀察板塊", list(SECTOR_LISTS.keys()))
    
    # 將選擇的板塊內容自動填入文字框
    default_text = SECTOR_LISTS[selected_sector]
    user_list = st.text_area("掃描名單 (可手動增減)", default_text, height=100)
    
    # 掃描按鈕區
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    scan_mode = None
    if col_b1.button("🔥 成交爆量"): scan_mode = 'vol'
    if col_b2.button("📈 漲幅強勢"): scan_mode = 'strong'
    if col_b3.button("📉 跌幅過重"): scan_mode = 'weak'
    if col_b4.button("🌊 觸底反彈"): scan_mode = 'rebound'
    
    if scan_mode or st.button("🚀 立即執行掃描"):
        tickers = [x.strip() for x in user_list.replace("\n", ",").split(",") if x]
        res = []
        
        # 進度條
        bar = st.progress(0, f"正在掃描 {len(tickers)} 檔股票...")
        
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            d = fetch_stock_data(t)
            if d:
                dist_low = (d['price'] - d['low_52']) / d['low_52'] * 100
                pe_disp = f"{d['pe']:.1f}" if d['pe'] else "-"
                res.append({
                    "代號": t, "名稱": d.get('name', t), 
                    "現價": d['price'], "漲跌%": round(d['change_pct'], 2),
                    "成交量": d['volume'], "本益比": pe_disp,
                    "KD": f"{d['k']:.0f}/{d['d']:.0f}",
                    "距低點%": round(dist_low, 1)
                })
        bar.empty()
        
        df = pd.DataFrame(res)
        if not df.empty:
            if scan_mode == 'vol': df = df.sort_values("成交量", ascending=False).head(10)
            elif scan_mode == 'strong': df = df.sort_values("漲跌%", ascending=False).head(10)
            elif scan_mode == 'weak': df = df.sort_values("漲跌%", ascending=True).head(10)
            elif scan_mode == 'rebound': df = df.sort_values("距低點%", ascending=True).head(10)
            
            st.dataframe(df) # 不使用 use_container_width 以防報錯
        else:
            st.warning("⚠️ 查無資料，可能是股票代號錯誤或網路連線暫時中斷。")

elif page == "💰 庫存管理":
    st.title("💰 我的庫存")
    df_port = get_portfolio()
    
    if not df_port.empty:
        res = []
        total_mkt = 0
        total_cost = 0
        bar = st.progress(0, "計算市值中...")
        for i, row in df_port.iterrows():
            bar.progress((i+1)/len(df_port))
            d = fetch_stock_data(row['ticker'])
            curr = d['price'] if d else row['cost']
            
            mkt = curr * row['shares']
            cost = row['cost'] * row['shares']
            pnl = mkt - cost
            total_mkt += mkt
            total_cost += cost
            
            res.append({
                "代號": row['ticker'], "現價": curr, 
                "損益": int(pnl), "報酬率%": round((pnl/cost)*100, 2)
            })
        bar.empty()
        
        c1, c2 = st.columns(2)
        tot_pnl = total_mkt - total_cost
        c1.metric("總市值", f"${total_mkt:,.0f}")
        c2.metric("總損益", f"${tot_pnl:,.0f}", f"{(tot_pnl/total_cost)*100:.2f}%")
        
        st.dataframe(pd.DataFrame(res))
        
        d_ticker = st.selectbox("刪除持股", df_port['ticker'])
        if st.button("刪除"):
            delete_portfolio(d_ticker)
            st.rerun()
    else:
        st.info("目前無庫存")
