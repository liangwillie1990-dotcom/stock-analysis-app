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
st.set_page_config(page_title="Joymax 戰情室 V12", layout="wide", page_icon="💰")

# ==========================================
# 1. 資料庫層 (升級庫存邏輯)
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
        c.execute('''CREATE TABLE IF NOT EXISTS system_status
                     (key TEXT PRIMARY KEY, value TEXT)''')
        conn.commit()
        conn.close()
    except: pass

def get_cached_stock(ticker, ttl_minutes=60):
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

# 🔥 V12 核心升級：智慧加碼邏輯 (平均成本法)
def add_portfolio_smart(ticker, buy_price, buy_shares):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. 先檢查這檔股票是否已經在庫存裡
    c.execute("SELECT cost, shares FROM portfolio WHERE ticker=?", (ticker,))
    row = c.fetchone()
    
    if row:
        # 情況 A: 已經有庫存 -> 執行「平均成本」計算
        old_cost, old_shares = row
        
        # 計算總投入成本
        total_cost = (old_cost * old_shares) + (buy_price * buy_shares)
        total_shares = old_shares + buy_shares
        
        # 算出新的平均成本
        new_avg_cost = total_cost / total_shares
        
        c.execute("UPDATE portfolio SET cost=?, shares=? WHERE ticker=?", 
                  (new_avg_cost, total_shares, ticker))
        
        msg = f"✅ 已加碼 {ticker}！\n舊成本 {old_cost:.1f} -> 新平均成本 {new_avg_cost:.1f}"
        
    else:
        # 情況 B: 新股票 -> 直接新增
        c.execute("INSERT INTO portfolio (ticker, cost, shares) VALUES (?, ?, ?)", 
                  (ticker, buy_price, buy_shares))
        msg = f"✅ 已新增 {ticker} 到庫存！"
        
    conn.commit()
    conn.close()
    return msg

def delete_portfolio(ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    conn.commit()
    conn.close()

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
# 2. 雙引擎抓取 (Twstock + Yahoo)
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
    if is_tw_stock:
        try:
            stock_id = ticker.replace(".TW", "").replace(".TWO", "")
            real = twstock.realtime.get(stock_id)
            if real['success']:
                data['price'] = float(real['realtime']['latest_trade_price'])
                data['name'] = real['info']['name']
        except: pass

    try:
        stock = yf.Ticker(yahoo_ticker)
        hist = stock.history(period="6mo")
        if hist.empty: return None

        if 'price' not in data: data['price'] = hist['Close'].iloc[-1]
            
        close = hist['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsv = (close - hist['Low'].rolling(9).min()) / (hist['High'].rolling(9).max() - hist['Low'].rolling(9).min()) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()

        change_pct = (data['price'] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100

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
            pe_s = hist['Close'] / eps
            valuation = {"cheap": eps*pe_s.min(), "fair": eps*pe_s.mean(), "expensive": eps*pe_s.max()}

        data.update({
            "change_pct": change_pct, "volume": hist['Volume'].iloc[-1],
            "pe": pe, "eps": eps, "yield": yield_val,
            "k": k.iloc[-1], "d": d.iloc[-1], "rsi": rsi.iloc[-1],
            "ma20": close.rolling(20).mean().iloc[-1],
            "history_close": hist['Close'].to_json(),
            "valuation": valuation, "low_52": hist['Low'].min()
        })
        save_to_cache(yahoo_ticker, data)
        return data
    except: return None

# ==========================================
# 3. 自動更新排程
# ==========================================
def auto_update_job():
    df_port = get_portfolio()
    targets = df_port['ticker'].tolist() + ["^TWII", "^TWOII", "^SOX", "^IXIC"]
    for t in targets:
        fetch_stock_data(t, use_cache=False)
        time.sleep(1)
    set_system_status("last_auto_update", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def run_scheduler():
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == "07:30":
            last = get_system_status("last_auto_update")
            if not last or not last.startswith(now.strftime("%Y-%m-%d")):
                auto_update_job()
        time.sleep(60)

@st.cache_resource
def start_background_thread():
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    return t

start_background_thread()

def check_daily_update_status():
    last = get_system_status("last_auto_update")
    now = datetime.now()
    if now.hour >= 8 and (not last or datetime.strptime(last, "%Y-%m-%d %H:%M:%S").date() < now.date()):
        st.toast("🚀 啟動早盤自動更新...", icon="🤖")
        auto_update_job()
        st.toast("✅ 資料已更新完畢", icon="✅")

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
# 4. UI 介面
# ==========================================
check_daily_update_status()

with st.sidebar:
    st.title("Joymax V12 智慧庫存")
    page = st.radio("功能", ["📊 戰情儀表板", "🚀 戰術掃描", "💰 庫存管理"])
    
    last_update = get_system_status("last_auto_update")
    st.caption(f"🕒 上次更新：{last_update if last_update else '等待中...'}")
    if st.button("🔄 立即刷新全站"):
        auto_update_job()
        st.rerun()

    # --- 側邊欄：新增/加碼區 (V12 更新重點) ---
    if page == "💰 庫存管理":
        st.divider()
        st.subheader("➕ 新增 / 加碼持股")
        st.info("💡 系統會自動合併股數並計算「平均成本」")
        
        t = st.text_input("股票代號", "2330").upper()
        # V12 更新：移除 1000 限制，改成 1 (支援零股)
        c = st.number_input("本次買入單價", min_value=0.0, value=1000.0, step=0.5)
        s = st.number_input("本次買入股數 (零股可)", min_value=1, value=1000, step=1)
        
        if st.button("💾 確認存入"):
            if not t.endswith("TW") and not t.endswith("TWO") and t[:2].isdigit():
                t += ".TW"
            # 呼叫新的智慧加碼函式
            msg = add_portfolio_smart(t, c, s)
            st.success(msg)
            time.sleep(1.5)
            st.rerun()

if page == "📊 戰情儀表板":
    st.title("📊 市場總覽")
    cols = st.columns(4)
    for i, (k, v) in enumerate({"^TWII":"加權","^TWOII":"櫃買","^SOX":"費半","^IXIC":"那指"}.items()):
        with cols[i]:
            d = fetch_stock_data(k)
            st.metric(v, f"{d['price']:,.0f}" if d else "N/A", f"{d['change_pct']:.2f}%" if d else "0%")
    st.divider()
    
    ticker = st.text_input("代號", "2330.TW").upper()
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
    c1, c2, c3, c4 = st.columns(4)
    mode = None
    if c1.button("🔥 爆量"): mode = 'vol'
    if c2.button("📈 強勢"): mode = 'strong'
    if c3.button("📉 弱勢"): mode = 'weak'
    if c4.button("🌊 反彈"): mode = 'rebound'
    
    user_list = st.text_area("名單", "2330, 2317, 2603, 3231, 0050, 0056, 2454, 2881")
    if mode or st.button("執行"):
        res = []
        ts = [x.strip() for x in user_list.split(",")]
        bar = st.progress(0, "掃描中...")
        for i, t in enumerate(ts):
            bar.progress((i+1)/len(ts))
            d = fetch_stock_data(t)
            if d:
                res.append({
                    "代號": t, "名稱": d.get('name',t), "現價": d['price'],
                    "漲跌%": round(d['change_pct'], 2), "成交量": d['volume'],
                    "本益比": f"{d['pe']:.1f}" if d['pe'] else "-", 
                    "距低點%": round((d['price']-d['low_52'])/d['low_52']*100, 1)
                })
        bar.empty()
        df = pd.DataFrame(res)
        if not df.empty:
            if mode == 'vol': df = df.sort_values("成交量", ascending=False)
            elif mode == 'strong': df = df.sort_values("漲跌%", ascending=False)
            elif mode == 'weak': df = df.sort_values("漲跌%", ascending=True)
            elif mode == 'rebound': df = df.sort_values("距低點%", ascending=True)
            st.dataframe(df.head(10), use_container_width=True)

elif page == "💰 庫存管理":
    st.title("💰 我的庫存 (支援零股與分批買入)")
    df_port = get_portfolio()
    
    if not df_port.empty:
        res = []
        tot_mkt = 0
        tot_cost = 0
        bar = st.progress(0, "計算市值...")
        for i, row in df_port.iterrows():
            bar.progress((i+1)/len(df_port))
            d = fetch_stock_data(row['ticker'])
            curr = d['price'] if d else row['cost']
            
            mkt = curr * row['shares']
            cost = row['cost'] * row['shares']
            tot_mkt += mkt
            tot_cost += cost
            
            res.append({
                "代號": row['ticker'], "持有股數": row['shares'],
                "平均成本": round(row['cost'], 2), "現價": curr,
                "損益": int(mkt - cost), 
                "報酬率%": round((mkt - cost)/cost*100, 2)
            })
        bar.empty()
        
        c1, c2 = st.columns(2)
        c1.metric("總市值", f"${tot_mkt:,.0f}")
        c2.metric("總損益", f"${tot_mkt-tot_cost:,.0f}", f"{(tot_mkt-tot_cost)/tot_cost*100:.2f}%")
        st.dataframe(pd.DataFrame(res), use_container_width=True)
        
        d_t = st.selectbox("刪除持股", df_port['ticker'])
        if st.button("刪除"):
            delete_portfolio(d_t)
            st.rerun()
    else:
        st.info("目前無庫存，請從左側新增。")
