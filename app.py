import streamlit as st
import yfinance as yf
import twstock
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
import time
import json
import threading
from datetime import datetime, timedelta

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 智動化戰情室 V11", layout="wide", page_icon="⏰")

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
        # 新增一個表來記錄系統自動更新的狀態
        c.execute('''CREATE TABLE IF NOT EXISTS system_status
                     (key TEXT PRIMARY KEY, value TEXT)''')
        conn.commit()
        conn.close()
    except: pass

def get_cached_stock(ticker, ttl_minutes=60): # 預設快取 60 分鐘
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT data, updated_at FROM stock_cache WHERE ticker=?", (ticker,))
        row = c.fetchone()
        conn.close()
        if row:
            data_str, updated_at_str = row
            # 判斷是否過期
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

# 系統狀態存取
def set_system_status(key, value):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("REPLACE INTO system_status (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
        conn.close()
    except: pass

def get_system_status(key):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT value FROM system_status WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    except: return None

init_db()

# ==========================================
# 2. 核心抓取引擎 (維持 V10 強大功能)
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
    # A. Twstock
    if is_tw_stock:
        try:
            stock_id = ticker.replace(".TW", "").replace(".TWO", "")
            real = twstock.realtime.get(stock_id)
            if real['success']:
                data['price'] = float(real['realtime']['latest_trade_price'])
                data['name'] = real['info']['name']
        except: pass

    # B. Yahoo
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
            valuation = {
                "cheap": eps * pe_series.min(),
                "fair": eps * pe_series.mean(),
                "expensive": eps * pe_series.max()
            }

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
    except: return None

# ==========================================
# 3. 自動化排程系統 (V11 新增核心)
# ==========================================
def auto_update_job():
    """執行自動更新任務"""
    print(f"[{datetime.now()}] 啟動自動更新排程...")
    df_port = get_portfolio()
    if df_port.empty:
        print("庫存為空，跳過更新")
        return
    
    tickers = df_port['ticker'].tolist()
    # 這裡我們加上大盤指數，確保儀表板也是新的
    tickers.extend(["^TWII", "^TWOII", "^SOX", "^IXIC"])
    
    for t in tickers:
        print(f"自動更新中: {t}")
        # 強制 use_cache=False 以獲取最新數據並寫入資料庫
        fetch_stock_data(t, use_cache=False)
        time.sleep(1) # 溫柔一點，避免被擋
        
    # 記錄更新時間
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_system_status("last_auto_update", now_str)
    print(f"自動更新完成: {now_str}")

def run_scheduler():
    """背景執行緒：每分鐘檢查一次時間"""
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # 設定觸發時間 07:30
        if current_time == "07:30":
            # 檢查今天是否已經跑過了 (避免 07:30 這一分鐘內重複跑)
            last_run = get_system_status("last_auto_update")
            if last_run and last_run.startswith(now.strftime("%Y-%m-%d")):
                pass # 今天跑過了
            else:
                auto_update_job()
        
        time.sleep(60) # 每分鐘檢查一次

# 啟動背景執行緒 (只啟動一次，避免重複)
@st.cache_resource
def start_background_thread():
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    return t

# 啟動它！
start_background_thread()

# 智慧補償檢查 (當使用者打開 App 時，檢查今天更新了沒)
def check_daily_update_status():
    last_run = get_system_status("last_auto_update")
    now = datetime.now()
    
    # 如果現在已經超過 07:30，且今天還沒跑過更新
    today_730 = now.replace(hour=7, minute=30, second=0, microsecond=0)
    
    needs_update = False
    if now > today_730:
        if not last_run:
            needs_update = True
        else:
            last_date = datetime.strptime(last_run, "%Y-%m-%d %H:%M:%S")
            if last_date.date() < now.date():
                needs_update = True
    
    if needs_update:
        st.toast("🚀 檢測到今日尚未更新，正在背景執行自動更新...", icon="🤖")
        # 這裡我們用一個簡單的迴圈在前景跑，讓使用者知道
        # 為了不卡住太久，這裡只更新庫存，不更新大盤
        df_p = get_portfolio()
        if not df_p.empty:
            for t in df_p['ticker'].tolist():
                fetch_stock_data(t, use_cache=False)
        set_system_status("last_auto_update", now.strftime("%Y-%m-%d %H:%M:%S"))
        st.toast("✅ 自動補償更新完成！", icon="✅")

# ==========================================
# 4. AI 報告
# ==========================================
def generate_ai_report(ticker, d):
    ta = []
    if d['k'] > d['d']: ta.append("KD金叉")
    else: ta.append("KD死叉")
    val_str = "合理"
    if d['pe']:
        if d['pe'] < 15: val_str = "低估"
        elif d['pe'] > 20: val_str = "偏高"
    return f"【AI日報】{d.get('name', ticker)}\n💰 {d['price']:.1f} ({d['change_pct']:+.2f}%)\n📊 {', '.join(ta)} | PE: {d['pe']:.1f}x ({val_str})"

# ==========================================
# 5. UI 介面
# ==========================================
check_daily_update_status() # 進頁面時先檢查

with st.sidebar:
    st.title("Joymax V11 智動版")
    page = st.radio("功能", ["📊 戰情儀表板", "🚀 戰術掃描", "💰 庫存管理"])
    
    # 顯示上次自動更新時間
    last_update = get_system_status("last_auto_update")
    if last_update:
        st.caption(f"🕒 上次自動更新：\n{last_update}")
    else:
        st.caption("🕒 等待 07:30 自動更新...")
        
    if st.button("🔄 立即手動全更新"):
        auto_update_job()
        st.rerun()

    if page == "💰 庫存管理":
        st.subheader("新增庫存")
        t = st.text_input("代號", "2330")
        c = st.number_input("成本", 1000.0)
        s = st.number_input("股數", 1000)
        if st.button("儲存"):
            add_portfolio(t, c, s)
            st.rerun()

if page == "📊 戰情儀表板":
    st.title("📊 市場總覽")
    cols = st.columns(4)
    indices = {"^TWII": "加權", "^TWOII": "櫃買", "^SOX": "費半", "^IXIC": "那指"}
    for i, (k, v) in enumerate(indices.items()):
        with cols[i]:
            d = fetch_stock_data(k)
            if d: st.metric(v, f"{d['price']:,.0f}", f"{d['change_pct']:.2f}%")
            else: st.metric(v, "N/A")
    st.divider()
    
    ticker = st.text_input("個股代號", "2330.TW").upper()
    d = fetch_stock_data(ticker)
    if d:
        st.subheader(f"📌 {d.get('name', ticker)}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{d['price']}", f"{d['change_pct']:.2f}%")
        c2.metric("本益比", f"{d['pe']:.1f}x" if d['pe'] else "-")
        c3.metric("KD", f"{d['k']:.0f}/{d['d']:.0f}")
        c4.metric("殖利率", f"{d['yield']:.2f}%")
        
        if d.get('valuation'):
            val = d['valuation']
            st.write("### 💎 估價分析")
            v1, v2, v3 = st.columns(3)
            v1.metric("便宜", f"{val['cheap']:.1f}")
            v2.metric("合理", f"{val['fair']:.1f}")
            v3.metric("昂貴", f"{val['expensive']:.1f}")
            
        st.line_chart(pd.read_json(d['history_close'], typ='series'))

elif page == "🚀 戰術掃描":
    st.title("🚀 快捷掃描")
    st.write("快速篩選：")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    scan_mode = None
    if col_b1.button("🔥 爆量"): scan_mode = 'vol'
    if col_b2.button("📈 強勢"): scan_mode = 'strong'
    if col_b3.button("📉 弱勢"): scan_mode = 'weak'
    if col_b4.button("🌊 反彈"): scan_mode = 'rebound'

    default = "2330, 2317, 2603, 3231, 0050, 0056, 2454, 2881"
    user_list = st.text_area("掃描名單", default)
    
    if scan_mode or st.button("執行"):
        ts = [x.strip() for x in user_list.replace("\n", ",").split(",") if x]
        res = []
        bar = st.progress(0, "掃描中...")
        for i, t in enumerate(ts):
            bar.progress((i+1)/len(ts))
            d = fetch_stock_data(t)
            if d:
                dist_low = (d['price'] - d['low_52']) / d['low_52'] * 100
                res.append({
                    "代號": t, "名稱": d.get('name', t), "現價": d['price'],
                    "漲跌%": round(d['change_pct'], 2), "成交量": d['volume'],
                    "本益比": f"{d['pe']:.1f}" if d['pe'] else "-", "距低點%": round(dist_low, 1)
                })
        bar.empty()
        df = pd.DataFrame(res)
        if not df.empty:
            if scan_mode == 'vol': df = df.sort_values("成交量", ascending=False)
            elif scan_mode == 'strong': df = df.sort_values("漲跌%", ascending=False)
            elif scan_mode == 'weak': df = df.sort_values("漲跌%", ascending=True)
            elif scan_mode == 'rebound': df = df.sort_values("距低點%", ascending=True)
            st.dataframe(df.head(10), use_container_width=True)

elif page == "💰 庫存管理":
    st.title("💰 我的庫存 (自動更新監控中)")
    df_port = get_portfolio()
    if not df_port.empty:
        res = []
        tot_mkt = 0
        tot_cost = 0
        for i, row in df_port.iterrows():
            d = fetch_stock_data(row['ticker']) # 這裡會直接讀取早上7:30更新好的快取
            curr = d['price'] if d else row['cost']
            mkt = curr * row['shares']
            cost = row['cost'] * row['shares']
            tot_mkt += mkt
            tot_cost += cost
            res.append({
                "代號": row['ticker'], "現價": curr, 
                "損益": int(mkt - cost), 
                "報酬率%": round((mkt - cost)/cost*100, 2),
                "上次更新": "✅ 已快取" if d else "⚠️ 待更新"
            })
        
        c1, c2 = st.columns(2)
        c1.metric("總市值", f"${tot_mkt:,.0f}")
        c2.metric("總損益", f"${tot_mkt-tot_cost:,.0f}", f"{(tot_mkt-tot_cost)/tot_cost*100:.2f}%")
        st.dataframe(pd.DataFrame(res), use_container_width=True)
        
        d_ticker = st.selectbox("刪除持股", df_port['ticker'])
        if st.button("刪除"):
            delete_portfolio(d_ticker)
            st.rerun()
    else:
        st.info("目前無庫存")
