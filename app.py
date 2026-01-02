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
st.set_page_config(page_title="Joymax 旗艦戰情室 V8.1", layout="wide", page_icon="🚀")

# ==========================================
# 1. 資料庫層 (SQLite)
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
    except Exception as e:
        st.error(f"資料庫初始化失敗: {e}")

def get_cached_stock(ticker, ttl_minutes=60):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT data, updated_at FROM stock_cache WHERE ticker=?", (ticker,))
        row = c.fetchone()
        conn.close()
        
        if row:
            data_str, updated_at_str = row
            updated_at = datetime.fromisoformat(updated_at_str)
            if datetime.now() - updated_at < timedelta(minutes=ttl_minutes):
                return json.loads(data_str)
    except:
        pass
    return None

def save_to_cache(ticker, data_dict):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        data_str = json.dumps(data_dict)
        c.execute("REPLACE INTO stock_cache (ticker, data, updated_at) VALUES (?, ?, ?)", 
                  (ticker, data_str, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except:
        pass

def add_portfolio(ticker, cost, shares):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("REPLACE INTO portfolio (ticker, cost, shares) VALUES (?, ?, ?)", (ticker, cost, shares))
    conn.commit()
    conn.close()

def get_portfolio():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM portfolio", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def delete_portfolio(ticker):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. 技術指標計算引擎
# ==========================================
def calculate_ta(df):
    try:
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
    except:
        # 如果計算失敗，填入預設值
        df['K'] = 50
        df['D'] = 50
        df['MACD'] = 0
        df['Signal'] = 0
        df['RSI'] = 50
    return df

# ==========================================
# 3. 核心數據抓取 (V8.1 強固版)
# ==========================================
def fetch_stock_data(ticker, use_cache=True):
    ticker = ticker.strip().upper()
    # 自動補 .TW (針對數字代號)
    if ticker.isdigit(): ticker += ".TW"
    
    if use_cache:
        cached = get_cached_stock(ticker)
        if cached: return cached

    try:
        stock = yf.Ticker(ticker)
        # 縮短抓取範圍以加快速度
        hist = stock.history(period="6mo")
        
        if hist.empty:
            # 再次嘗試不加 .TW (針對美股或指數)
            stock = yf.Ticker(ticker.replace(".TW", ""))
            hist = stock.history(period="6mo")
            if hist.empty: return None

        # 計算技術指標
        hist = calculate_ta(hist)
        
        # 基礎數據 (絕對不會失敗的部分)
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        change_pct = (current - prev) / prev * 100
        volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
        
        # 進階數據 (info 容易失敗，需獨立處理)
        info = {}
        try:
            info = stock.info
        except:
            pass # 抓不到 info 就跳過，不要讓整個程式崩潰

        eps = info.get('trailingEps') or info.get('forwardEps')
        pe = None
        if eps:
            pe = current / eps
        
        # 容錯處理：若 info 裡沒名字，就用代號
        name = info.get('longName', ticker)
        
        data = {
            "price": current,
            "change_pct": change_pct,
            "volume": volume,
            "eps": eps,
            "pe": pe,
            "yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "ma20": hist['Close'].rolling(20).mean().iloc[-1] if len(hist)>20 else current,
            "ma60": hist['Close'].rolling(60).mean().iloc[-1] if len(hist)>60 else current,
            "k": hist['K'].iloc[-1],
            "d": hist['D'].iloc[-1],
            "macd": hist['MACD'].iloc[-1],
            "macd_sig": hist['Signal'].iloc[-1],
            "rsi": hist['RSI'].iloc[-1],
            "history_close": hist['Close'].to_json(),
            "name": name
        }
        
        save_to_cache(ticker, data)
        return data

    except Exception as e:
        # 如果真的發生不可預期的錯誤，印出來除錯，但不崩潰
        print(f"Fetch Error {ticker}: {e}")
        return None

# ==========================================
# 4. AI 解盤引擎
# ==========================================
def generate_ai_report(ticker, d):
    date_str = datetime.now().strftime("%Y/%m/%d")
    
    ta_signal = []
    if d['k'] > d['d']: ta_signal.append("KD黃金交叉(偏多)")
    else: ta_signal.append("KD死亡交叉(偏空)")
    if d['rsi'] > 70: ta_signal.append("RSI過熱")
    elif d['rsi'] < 30: ta_signal.append("RSI超賣")
    if not ta_signal: ta_signal.append("技術面盤整")
    
    ta_str = "、".join(ta_signal)
    yield_str = f"{d['yield']:.2f}%" if d['yield'] > 0 else "N/A"
    pe_str = f"{d['pe']:.1f}倍" if d['pe'] else "N/A"
    
    full_text = f"""
【Joymax 智能投顧】{d['name']} ({ticker})
📅 {date_str} | 💰 收盤：{d['price']:.1f} ({d['change_pct']:+.2f}%)
📊 殖利率：{yield_str} | 本益比：{pe_str}

🤖 AI 解析：
1. 技術面：{ta_str}。
2. 趨勢：股價{"站上" if d['price'] > d['ma20'] else "跌破"}月線。
3. 建議：{"技術面轉強，可留意佈局點" if d['change_pct']>0 else "短線修正，建議觀察支撐"}。
    """
    return full_text.strip()

# ==========================================
# 5. UI 介面
# ==========================================
with st.sidebar:
    st.title("Joymax V8.1 強固版")
    page = st.radio("前往頁面", ["📊 戰情儀表板", "💰 我的庫存管理", "🚀 戰術掃描"])
    st.markdown("---")
    
    if page == "💰 我的庫存管理":
        st.subheader("新增持股")
        p_ticker = st.text_input("代號", "2330").upper()
        p_cost = st.number_input("成本", value=1000.0)
        p_shares = st.number_input("股數", value=1000, step=100)
        if st.button("💾 儲存"):
            if not p_ticker.endswith("TW"): p_ticker += ".TW"
            add_portfolio(p_ticker, p_cost, p_shares)
            st.success("已儲存")
            time.sleep(0.5)
            st.rerun()

if page == "📊 戰情儀表板":
    st.title("📊 市場總覽")
    
    # 指數區塊 (這裡最容易卡住，V8.1 已做容錯)
    cols = st.columns(4)
    # 註：美股指數代號不需 .TW，程式會自動判斷
    indices = {"^TWII": "加權指數", "^TWOII": "櫃買指數", "^SOX": "費半指數", "^IXIC": "那斯達克"}
    
    for i, (k, v) in enumerate(indices.items()):
        with cols[i]:
            d = fetch_stock_data(k)
            if d: 
                st.metric(v, f"{d['price']:,.0f}", f"{d['change_pct']:.2f}%")
            else: 
                st.metric(v, "N/A", "無資料")

    st.divider()
    
    # 個股深度分析
    col_input, col_btn = st.columns([3, 1])
    ticker = col_input.text_input("輸入個股代號", "2330.TW").upper()
    if col_btn.button("🔍 分析"):
        d = fetch_stock_data(ticker, use_cache=False)
    else:
        d = fetch_stock_data(ticker)

    if d:
        st.subheader(f"📌 {d['name']} ({ticker})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{d['price']}", f"{d['change_pct']:.2f}%")
        c2.metric("KD", f"{d['k']:.0f}/{d['d']:.0f}")
        c3.metric("RSI", f"{d['rsi']:.1f}")
        c4.metric("殖利率", f"{d['yield']:.2f}%")
        
        with st.expander("🤖 AI 報告 (點擊展開)", expanded=True):
            st.code(generate_ai_report(ticker, d), language="text")
            
        hist_series = pd.read_json(d['history_close'], typ='series')
        st.line_chart(hist_series)
    else:
        st.error(f"無法取得 {ticker} 的資料，請檢查代號是否正確。")

elif page == "💰 我的庫存管理":
    st.title("💰 庫存管理")
    df_port = get_portfolio()
    
    if not df_port.empty:
        total_mkt = 0
        total_cost = 0
        res_list = []
        
        # 顯示進度條
        bar = st.progress(0, "更新庫存現價中...")
        for i, row in df_port.iterrows():
            bar.progress((i+1)/len(df_port))
            d = fetch_stock_data(row['ticker'])
            curr = d['price'] if d else row['cost']
            
            mkt = curr * row['shares']
            cost = row['cost'] * row['shares']
            pnl = mkt - cost
            
            total_mkt += mkt
            total_cost += cost
            
            res_list.append({
                "代號": row['ticker'], "股數": row['shares'],
                "成本": row['cost'], "現價": curr,
                "市值": int(mkt), "損益": int(pnl),
                "報酬率%": round((pnl/cost)*100, 2)
            })
        bar.empty()
        
        c1, c2 = st.columns(2)
        tot_pnl = total_mkt - total_cost
        c1.metric("總市值", f"${total_mkt:,.0f}")
        c2.metric("總損益", f"${tot_pnl:,.0f}", f"{(tot_pnl/total_cost)*100:.2f}%")
        
        st.dataframe(pd.DataFrame(res_list), use_container_width=True)
        
        del_t = st.selectbox("刪除代號", df_port['ticker'])
        if st.button("刪除"):
            delete_portfolio(del_t)
            st.rerun()
    else:
        st.info("目前無庫存，請從側邊欄新增。")

elif page == "🚀 戰術掃描":
    st.title("🚀 快速掃描")
    # 內建一個不會太大的清單以免卡住
    default_list = "2330, 2317, 2454, 2603, 2881, 0050, 0056"
    user_list = st.text_area("輸入代號 (逗號分隔)", default_list)
    
    if st.button("執行掃描"):
        tickers = [x.strip() for x in user_list.replace("\n", ",").split(",") if x]
        res = []
        bar = st.progress(0, "掃描中...")
        
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            d = fetch_stock_data(t)
            if d:
                res.append({
                    "代號": t, "名稱": d['name'], "現價": d['price'],
                    "漲跌%": round(d['change_pct'], 2),
                    "KD": f"{d['k']:.0f}/{d['d']:.0f}",
                    "本益比": f"{d['pe']:.1f}" if d['pe'] else "-"
                })
        bar.empty()
        st.dataframe(pd.DataFrame(res), use_container_width=True)
