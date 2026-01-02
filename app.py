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
from datetime import datetime, timedelta

# ==========================================
# 0. 全局設定與 CSS 美化
# ==========================================
st.set_page_config(
    page_title="Joymax Titan V13",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# 注入自定義 CSS 以達到「APP 級」的視覺體驗
st.markdown("""
<style>
    /* 全局字體優化 */
    .stApp { font-family: 'Microsoft JhengHei', sans-serif; }
    
    /* 指標卡片美化 */
    div[data-testid="stMetric"] {
        background-color: #2b313e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        border: 1px solid #3d4452;
    }
    div[data-testid="stMetricLabel"] { color: #cfcfcf; }
    div[data-testid="stMetricValue"] { font-weight: bold; }
    
    /* 表格優化 */
    div[data-testid="stDataFrame"] { margin-top: 10px; }
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] {
        background-color: #1e2129;
    }
    
    /* 按鈕樣式 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    
    /* 頁籤樣式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫管理層 (Database Manager)
# ==========================================
DB_NAME = "joymax_titan.db"

class DBManager:
    """處理所有 SQLite 資料庫操作的類別"""
    
    @staticmethod
    def init_db():
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # 快取表
        c.execute('''CREATE TABLE IF NOT EXISTS stock_cache
                     (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
        # 庫存表
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                     (ticker TEXT PRIMARY KEY, cost REAL, shares INTEGER, group_name TEXT)''')
        # 自選股清單表
        c.execute('''CREATE TABLE IF NOT EXISTS watchlists
                     (list_name TEXT, tickers TEXT, PRIMARY KEY (list_name))''')
        # 系統狀態
        c.execute('''CREATE TABLE IF NOT EXISTS system_status
                     (key TEXT PRIMARY KEY, value TEXT)''')
        conn.commit()
        conn.close()
        
        # 初始化預設清單
        DBManager.init_default_lists()

    @staticmethod
    def init_default_lists():
        defaults = {
            "權值龍頭": "2330, 2317, 2454, 2308, 2881, 2882, 1301, 2002, 0050",
            "AI 供應鏈": "2330, 2317, 2382, 3231, 2357, 6669, 2379, 3035",
            "航運三雄": "2603, 2609, 2615, 2637, 5608",
            "金融存股": "2881, 2882, 2891, 2884, 2886, 2892, 5880",
            "高股息 ETF": "0056, 00878, 00919, 00929, 00713"
        }
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for name, tickers in defaults.items():
            c.execute("INSERT OR IGNORE INTO watchlists (list_name, tickers) VALUES (?, ?)", (name, tickers))
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
    def update_portfolio(ticker, price, shares):
        """智慧加碼：平均成本法"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT cost, shares FROM portfolio WHERE ticker=?", (ticker,))
        row = c.fetchone()
        
        if row:
            old_cost, old_shares = row
            total_cost = (old_cost * old_shares) + (price * shares)
            total_shares = old_shares + shares
            new_avg = total_cost / total_shares if total_shares > 0 else 0
            c.execute("UPDATE portfolio SET cost=?, shares=? WHERE ticker=?", (new_avg, total_shares, ticker))
            msg = f"加碼成功！新平均成本: {new_avg:.2f}"
        else:
            c.execute("INSERT INTO portfolio (ticker, cost, shares, group_name) VALUES (?, ?, ?, ?)", 
                      (ticker, price, shares, 'Default'))
            msg = "新增庫存成功！"
        
        conn.commit()
        conn.close()
        return msg

    @staticmethod
    def get_portfolio_df():
        try:
            conn = sqlite3.connect(DB_NAME)
            df = pd.read_sql("SELECT * FROM portfolio", conn)
            conn.close()
            return df
        except: return pd.DataFrame()

    @staticmethod
    def delete_portfolio(ticker):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
        conn.commit()
        conn.close()

    @staticmethod
    def get_watchlists():
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT list_name, tickers FROM watchlists")
        rows = c.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

# 初始化資料庫
DBManager.init_db()

# ==========================================
# 2. 技術分析引擎 (Technical Analysis Engine)
# ==========================================
class TAEngine:
    @staticmethod
    def calculate(df):
        if df.empty: return df
        
        # 移動平均
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
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
        df['Hist'] = df['MACD'] - df['Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林通道 (Bollinger Bands)
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Low'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        return df

    @staticmethod
    def get_signals(d):
        signals = []
        # KD 訊號
        if d['k'] > d['d'] and d['k'] < 30: signals.append("KD低檔金叉")
        elif d['k'] < d['d'] and d['k'] > 80: signals.append("KD高檔死叉")
        
        # 均線訊號
        if d['price'] > d['ma20'] and d['price'] > d['ma60']: signals.append("多頭排列")
        elif d['price'] < d['ma20'] and d['price'] < d['ma60']: signals.append("空頭排列")
        
        # MACD
        if d['macd'] > d['macd_sig']: signals.append("MACD翻紅")
        
        # 價格位置
        if d['price'] >= d['bb_up']: signals.append("突破布林上緣")
        elif d['price'] <= d['bb_low']: signals.append("跌破布林下緣")
        
        return signals

# ==========================================
# 3. 數據抓取引擎 (Data Fetcher) - 支援並行
# ==========================================
class DataFetcher:
    @staticmethod
    def clean_ticker(ticker):
        ticker = ticker.strip().upper()
        if ticker.isdigit(): ticker += ".TW"
        if not (ticker.endswith(".TW") or ticker.endswith(".TWO")) and ticker[:1].isdigit():
            ticker += ".TW"
        return ticker

    @staticmethod
    def fetch_single(ticker, use_cache=True):
        ticker = DataFetcher.clean_ticker(ticker)
        
        # 1. 查快取
        if use_cache:
            cached = DBManager.get_cache(ticker)
            if cached: return cached
            
        data = {}
        # 2. Twstock 抓即時 (僅限台股)
        is_tw = ticker[:2].isdigit()
        if is_tw:
            try:
                sid = ticker.replace(".TW", "").replace(".TWO", "")
                real = twstock.realtime.get(sid)
                if real['success']:
                    data['price'] = float(real['realtime']['latest_trade_price'])
                    data['name'] = real['info']['name']
            except: pass
            
        # 3. Yahoo 抓完整數據
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if hist.empty: return None
            
            # 技術指標計算
            hist = TAEngine.calculate(hist)
            
            # 如果 Twstock 沒抓到，用 Yahoo 補
            if 'price' not in data: data['price'] = hist['Close'].iloc[-1]
            if 'name' not in data: 
                try: data['name'] = stock.info.get('longName', ticker)
                except: data['name'] = ticker

            # 基礎數據
            last_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_pct = (last_close - prev_close) / prev_close * 100
            
            # 基本面 (EPS/PE)
            eps, pe, yield_val = None, None, 0
            try:
                info = stock.info
                eps = info.get('trailingEps') or info.get('forwardEps')
                if eps: pe = data['price'] / eps
                yield_val = info.get('dividendYield', 0) * 100
            except: pass
            
            # 本益比估價
            val = {}
            if eps:
                pe_series = hist['Close'] / eps
                val = {
                    "cheap": eps * pe_series.min(),
                    "fair": eps * pe_series.mean(),
                    "expensive": eps * pe_series.max()
                }

            # 組合數據包
            data.update({
                "ticker": ticker,
                "change_pct": change_pct,
                "volume": hist['Volume'].iloc[-1],
                "pe": pe, "eps": eps, "yield": yield_val,
                "k": hist['K'].iloc[-1], "d": hist['D'].iloc[-1],
                "rsi": hist['RSI'].iloc[-1],
                "macd": hist['MACD'].iloc[-1], "macd_sig": hist['Signal'].iloc[-1],
                "ma5": hist['MA5'].iloc[-1], "ma20": hist['MA20'].iloc[-1], "ma60": hist['MA60'].iloc[-1],
                "bb_up": hist['BB_Up'].iloc[-1], "bb_low": hist['BB_Low'].iloc[-1],
                "history_json": hist.reset_index().to_json(date_format='iso'), # 存完整歷史供圖表用
                "valuation": val,
                "high_52": hist['High'].max(), "low_52": hist['Low'].min()
            })
            
            DBManager.save_cache(ticker, data)
            return data
        except Exception as e:
            # print(f"Fetch Error {ticker}: {e}")
            return None

    @staticmethod
    def fetch_batch(tickers, max_workers=10):
        """並行抓取，極速模式"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(DataFetcher.fetch_single, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                if data: results.append(data)
        return results

# ==========================================
# 4. 後台自動排程 (Scheduler)
# ==========================================
def run_scheduler():
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == "07:30":
            # 每天早上執行一次全庫存更新
            df = DBManager.get_portfolio_df()
            if not df.empty:
                targets = df['ticker'].tolist()
                DataFetcher.fetch_batch(targets)
        time.sleep(60)

@st.cache_resource
def start_thread():
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    return t

start_thread()

# ==========================================
# 5. UI 介面組件 (UI Components)
# ==========================================

def render_gauge_chart(value, min_v, max_v, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_v, max_v]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_v, min_v + (max_v-min_v)*0.33], 'color': "lightgreen"},
                {'range': [min_v + (max_v-min_v)*0.33, min_v + (max_v-min_v)*0.66], 'color': "lightyellow"},
                {'range': [min_v + (max_v-min_v)*0.66, max_v], 'color': "salmon"}],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

def render_candle_chart(data):
    try:
        df = pd.read_json(data['history_json'])
        # 處理日期索引
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'index' in df.columns: # Sometimes reset_index creates 'index'
            df['index'] = pd.to_datetime(df['index'])
            df.set_index('index', inplace=True)
            
        # 繪圖
        fig = go.Figure()
        
        # K線
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
        ))
        
        # 均線
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='月線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='季線'))
        
        # 布林通道
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], fill='tonexty', fillcolor='rgba(0,0,255,0.05)', line=dict(width=0), name='布林帶'))

        fig.update_layout(
            title=f"{data['name']} ({data['ticker']}) 技術線圖",
            yaxis_title='股價', xaxis_rangeslider_visible=False,
            height=450, template="plotly_dark",
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"圖表繪製失敗: {e}")

# ==========================================
# 6. 主程式邏輯 (Main App Logic)
# ==========================================

# --- 側邊欄：導航與設定 ---
with st.sidebar:
    st.title("🏛️ Titan 戰情室")
    st.info(f"V13.0 企業旗艦版")
    
    # 全局功能
    if st.button("🔄 立即更新所有數據"):
        with st.spinner("正在啟動並行引擎更新全站數據..."):
            # 更新大盤、自選、庫存
            DataFetcher.fetch_batch(["^TWII", "^TWOII", "^SOX", "^IXIC"])
            df_p = DBManager.get_portfolio_df()
            if not df_p.empty: DataFetcher.fetch_batch(df_p['ticker'].tolist())
        st.success("更新完成")
        time.sleep(1)
        st.rerun()

    st.divider()
    
    # 庫存快手
    with st.expander("💰 庫存速記 (智慧加碼)", expanded=True):
        add_ticker = st.text_input("代號", "2330")
        add_price = st.number_input("價格", 0.0, step=0.5)
        add_shares = st.number_input("股數", 1, step=1)
        if st.button("存入庫存"):
            t = DataFetcher.clean_ticker(add_ticker)
            msg = DBManager.update_portfolio(t, add_price, add_shares)
            st.success(msg)
            time.sleep(1)
            st.rerun()

# --- 主頁面 Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 戰情儀表板", "🔍 策略選股雷達", "💰 資產管理中心", "📈 個股深度戰情"])

# --- Tab 1: 戰情儀表板 (Dashboard) ---
with tab1:
    st.subheader("🌍 全球市場與大盤")
    
    # 並行抓取指數
    indices = {"^TWII":"加權指數", "^TWOII":"櫃買指數", "^SOX":"費半指數", "^IXIC":"那斯達克"}
    idx_data = DataFetcher.fetch_batch(list(indices.keys()))
    
    # 顯示 Metrics
    cols = st.columns(4)
    for i, (k, v) in enumerate(indices.items()):
        # 找對應的資料
        d = next((x for x in idx_data if x['ticker'] == k), None)
        with cols[i]:
            if d:
                st.metric(
                    label=v, 
                    value=f"{d['price']:,.0f}", 
                    delta=f"{d['change_pct']:.2f}%"
                )
            else:
                st.metric(label=v, value="Loading...")
    
    st.divider()
    
    # 精選板塊輪動 (Sector Rotation)
    st.subheader("🏭 觀察清單板塊輪動")
    watchlists = DBManager.get_watchlists()
    selected_list = st.selectbox("選擇觀察板塊", list(watchlists.keys()))
    
    if selected_list:
        tickers = [t.strip() for t in watchlists[selected_list].split(",")]
        
        with st.spinner("🚀 Titan 引擎啟動：正在並行掃描板塊成分股..."):
            start_t = time.time()
            batch_data = DataFetcher.fetch_batch(tickers)
            end_t = time.time()
        
        # 整理成 DataFrame
        rows = []
        for d in batch_data:
            signals = TAEngine.get_signals(d)
            rows.append({
                "代號": d['ticker'], "名稱": d['name'], 
                "現價": d['price'], "漲跌%": d['change_pct'],
                "本益比": d['pe'] if d['pe'] else np.nan,
                "殖利率%": d['yield'],
                "KD": f"{d['k']:.0f}/{d['d']:.0f}",
                "訊號": ", ".join(signals) if signals else "盤整"
            })
            
        df_view = pd.DataFrame(rows)
        if not df_view.empty:
            st.caption(f"掃描耗時: {end_t - start_t:.2f} 秒")
            
            # 使用 column_config 視覺化
            st.dataframe(
                df_view.sort_values("漲跌%", ascending=False),
                column_config={
                    "漲跌%": st.column_config.NumberColumn(format="%.2f%%"),
                    "本益比": st.column_config.NumberColumn(format="%.1f"),
                    "殖利率%": st.column_config.NumberColumn(format="%.2f%%"),
                    "現價": st.column_config.NumberColumn(format="%.1f"),
                },
                use_container_width=True,
                hide_index=True
            )

# --- Tab 2: 策略選股雷達 (Screener) ---
with tab2:
    st.subheader("🎯 條件篩選器 (Screener)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("#### 🛠️ 設定篩選條件")
        scan_source = st.radio("掃描範圍", ["自訂清單", "全庫存", "權值龍頭"])
        
        filter_pe = st.slider("本益比低於", 10, 50, 20)
        filter_yield = st.slider("殖利率高於 (%)", 0.0, 10.0, 3.0)
        filter_kd_gold = st.checkbox("KD 黃金交叉 (K > D)", value=False)
        filter_bullish = st.checkbox("多頭排列 (價 > 月 > 季)", value=False)
        
        target_tickers = []
        if scan_source == "權值龍頭":
            target_tickers = watchlists["權值龍頭"].split(",")
        elif scan_source == "全庫存":
            df_p = DBManager.get_portfolio_df()
            if not df_p.empty: target_tickers = df_p['ticker'].tolist()
        else:
            raw = st.text_area("輸入代號 (逗號分隔)", "2330, 2317, 2454, 2603, 3231, 0050")
            target_tickers = raw.split(",")

        run_scan = st.button("🚀 開始篩選")

    with col2:
        if run_scan:
            clean_targets = [t.strip() for t in target_tickers if t.strip()]
            with st.spinner(f"正在分析 {len(clean_targets)} 檔標的..."):
                scan_res = DataFetcher.fetch_batch(clean_targets)
            
            filtered = []
            for d in scan_res:
                # 條件判斷
                is_match = True
                if d['pe'] and d['pe'] > filter_pe: is_match = False
                if d['yield'] < filter_yield: is_match = False
                if filter_kd_gold and not (d['k'] > d['d']): is_match = False
                if filter_bullish and not (d['price'] > d['ma20'] and d['price'] > d['ma60']): is_match = False
                
                if is_match:
                    filtered.append({
                        "代號": d['ticker'], "名稱": d['name'], "現價": d['price'],
                        "本益比": d['pe'], "殖利率": d['yield'], "KD": f"{d['k']:.0f}/{d['d']:.0f}",
                        "RSI": f"{d['rsi']:.1f}"
                    })
            
            st.markdown(f"#### 🔍 篩選結果 ({len(filtered)}/{len(scan_res)})")
            if filtered:
                st.dataframe(pd.DataFrame(filtered), use_container_width=True)
            else:
                st.warning("沒有符合條件的股票。")

# --- Tab 3: 資產管理中心 (Portfolio) ---
with tab3:
    st.subheader("💰 我的庫存損益")
    
    df_port = DBManager.get_portfolio_df()
    
    if df_port.empty:
        st.info("目前無庫存，請至側邊欄新增。")
    else:
        # 即時更新庫存現價
        tickers = df_port['ticker'].tolist()
        latest_data = DataFetcher.fetch_batch(tickers)
        data_map = {d['ticker']: d for d in latest_data}
        
        portfolio_rows = []
        total_market = 0
        total_cost = 0
        
        for idx, row in df_port.iterrows():
            d = data_map.get(row['ticker'])
            curr_price = d['price'] if d else row['cost']
            mkt_val = curr_price * row['shares']
            cost_val = row['cost'] * row['shares']
            pnl = mkt_val - cost_val
            pnl_pct = (pnl / cost_val) * 100 if cost_val > 0 else 0
            
            total_market += mkt_val
            total_cost += cost_val
            
            portfolio_rows.append({
                "代號": row['ticker'], "持有股數": row['shares'],
                "平均成本": row['cost'], "現價": curr_price,
                "市值": mkt_val, "損益": pnl, "報酬率%": pnl_pct
            })
            
        # 總覽 Metrics
        m1, m2, m3 = st.columns(3)
        tot_pnl = total_market - total_cost
        tot_pnl_pct = (tot_pnl / total_cost * 100) if total_cost > 0 else 0
        
        m1.metric("總資產市值", f"${total_market:,.0f}")
        m2.metric("總投入成本", f"${total_cost:,.0f}")
        m3.metric("未實現損益", f"${tot_pnl:,.0f}", f"{tot_pnl_pct:.2f}%")
        
        # 詳細清單
        df_view = pd.DataFrame(portfolio_rows)
        st.dataframe(
            df_view,
            column_config={
                "報酬率%": st.column_config.NumberColumn(format="%.2f%%"),
                "損益": st.column_config.NumberColumn(format="$%d"),
                "市值": st.column_config.NumberColumn(format="$%d"),
                "現價": st.column_config.NumberColumn(format="%.1f"),
                "平均成本": st.column_config.NumberColumn(format="%.1f"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # 刪除功能
        c1, c2 = st.columns([3, 1])
        with c2:
            del_target = st.selectbox("選擇刪除標的", df_port['ticker'])
            if st.button("🗑️ 刪除持股"):
                DBManager.delete_portfolio(del_target)
                st.rerun()

        # 資產圓餅圖
        fig = px.pie(df_view, values='市值', names='代號', title='資產配置分布', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: 個股深度戰情 (Deep Dive) ---
with tab4:
    st.subheader("📈 個股全方位分析")
    
    col_search, col_act = st.columns([3, 1])
    ticker_input = col_search.text_input("輸入代號", "2330.TW").upper()
    if col_act.button("🔍 深度分析"):
        # 強制更新該股
        DBManager.save_cache(DataFetcher.clean_ticker(ticker_input), {}) # 清空快取
    
    d = DataFetcher.fetch_single(ticker_input)
    
    if d:
        st.markdown(f"### {d['name']} ({d['ticker']})")
        
        # 1. 核心指標列
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("現價", d['price'], f"{d['change_pct']:.2f}%")
        k2.metric("本益比", f"{d['pe']:.1f}x" if d['pe'] else "N/A")
        k3.metric("殖利率", f"{d['yield']:.2f}%")
        k4.metric("KD值", f"{d['k']:.0f}/{d['d']:.0f}")
        k5.metric("RSI", f"{d['rsi']:.1f}")
        
        # 2. 技術圖表區
        render_candle_chart(d)
        
        # 3. 估值儀表板與 AI 建議
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### 💎 價值評估")
            if d.get('valuation'):
                val = d['valuation']
                # 繪製儀表圖
                render_gauge_chart(d['price'], val['cheap']*0.8, val['expensive']*1.2, "股價位階")
                st.info(f"便宜: {val['cheap']:.1f} | 合理: {val['fair']:.1f} | 昂貴: {val['expensive']:.1f}")
            else:
                st.warning("無 EPS 數據，無法進行估值計算")
                
        with c2:
            st.markdown("#### 🤖 泰坦 AI 綜合點評")
            signals = TAEngine.get_signals(d)
            signal_color = "green" if any("金叉" in s or "多頭" in s for s in signals) else "red"
            
            html_signals = "".join([f"<span style='background:#333;padding:5px;border-radius:5px;margin:2px;border:1px solid #555'>{s}</span>" for s in signals])
            
            st.markdown(f"""
            <div style="background-color:#262730; padding:20px; border-radius:10px; border-left: 5px solid {signal_color}">
                <h5>技術訊號偵測</h5>
                {html_signals if signals else "目前無明顯趨勢訊號"}
                <hr>
                <h5>操作建議</h5>
                若為長線投資者，建議參考左側估值儀表板，於綠色區間分批佈局。
                若為短線交易者，請關注上方技術訊號與成交量變化。
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("查無資料，請檢查代號。")
