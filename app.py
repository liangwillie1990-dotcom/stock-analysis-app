"""
Willie's Alpha V15.1 - Institutional Grade Quant System (Fixed Dependencies)
Author: Gemini AI
Description:
    An advanced stock analysis platform featuring:
    1. AI Scoring System (0-100)
    2. Monte Carlo Simulation for Risk Analysis (Requires scipy)
    3. Technical Pattern Recognition
    4. Portfolio & Ledger Management
    5. Hybrid Data Fetching (Twstock + Yahoo)
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
import random
from datetime import datetime, timedelta

# 科學運算模組 (本次錯誤修正的關鍵)
from scipy.stats import norm

# ==========================================
# 0. 全局設定與 CSS 視覺系統 (Visual System)
# ==========================================
st.set_page_config(
    page_title="Willie's Alpha V15.1",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# 注入駭客任務/彭博終端機風格 CSS
st.markdown("""
<style>
    /* 全局變數定義 */
    :root {
        --primary-color: #00d4ff;
        --bull-color: #00fa9a;
        --bear-color: #ff4d4d;
        --bg-dark: #0e1117;
        --card-bg: #1a1c24;
        --text-gray: #a0a0a0;
    }
    
    /* 字體優化 */
    .stApp { font-family: 'Roboto Mono', 'Microsoft JhengHei', monospace; background-color: var(--bg-dark); }
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] {
        background-color: #111;
        border-right: 1px solid #333;
    }
    
    /* 數據卡片 (Metrics) - 玻璃擬態風格 */
    div[data-testid="stMetric"] {
        background: rgba(26, 28, 36, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: var(--primary-color);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.2);
    }
    
    /* 標籤頁 (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 6px;
        color: var(--text-gray);
        border: none;
        font-weight: 600;
        transition: color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333;
        color: var(--primary-color);
    }
    
    /* 按鈕特效 */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        font-weight: bold;
        letter-spacing: 1px;
        transition: all 0.3s;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.6);
        transform: scale(1.02);
    }
    
    /* 表格樣式 */
    div[data-testid="stDataFrame"] {
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* 自定義進度條 */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 資料庫管理層 (Database Layer)
# ==========================================
DB_NAME = "willie_alpha.db"

class DBManager:
    """處理 SQLite 所有操作的單例模式"""
    
    @staticmethod
    def init_db():
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            
            # 快取表 (包含詳細的 JSON 數據)
            c.execute('''CREATE TABLE IF NOT EXISTS stock_cache
                         (ticker TEXT PRIMARY KEY, data TEXT, updated_at TIMESTAMP)''')
            
            # 交易流水帳 (Ledger)
            c.execute('''CREATE TABLE IF NOT EXISTS transactions
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          date TIMESTAMP,
                          ticker TEXT,
                          type TEXT,
                          price REAL,
                          shares INTEGER,
                          amount REAL,
                          fee REAL,
                          note TEXT)''')
                          
            # 庫存總表 (Portfolio)
            c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                         (ticker TEXT PRIMARY KEY, avg_cost REAL, shares INTEGER, group_name TEXT)''')
            
            # 系統設定 (System Config)
            c.execute('''CREATE TABLE IF NOT EXISTS system_config
                         (key TEXT PRIMARY KEY, value TEXT)''')
                         
            conn.commit()
            conn.close()
            DBManager._seed_defaults()
        except Exception as e:
            st.error(f"資料庫初始化失敗: {e}")

    @staticmethod
    def _seed_defaults():
        """寫入預設的觀察清單"""
        defaults = {
            "list_tech": "2330,2317,2454,2308,3231,2382,6669,3443,2357,2379",
            "list_finance": "2881,2882,2891,5880,2886,2892,2884,2890",
            "list_shipping": "2603,2609,2615,2637,5608,2618,2610",
            "list_etf": "0050,0056,00878,00919,00929,00713,00940"
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
    def record_transaction(ticker, trans_type, price, shares, note="Manual"):
        """記錄交易並更新庫存 (含手續費計算)"""
        date = datetime.now()
        amount = price * shares
        # 台股手續費 0.1425% (低消20暫不計), 交易稅 0.3%
        fee = int(amount * 0.001425)
        tax = int(amount * 0.003) if trans_type == 'SELL' else 0
        total_cash_flow = amount + fee if trans_type == 'BUY' else amount - fee - tax
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # 1. 寫入流水帳
        c.execute('''INSERT INTO transactions (date, ticker, type, price, shares, amount, fee, note)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (date, ticker, trans_type, price, shares, total_cash_flow, fee+tax, note))
        
        # 2. 更新庫存邏輯 (平均成本法)
        c.execute("SELECT avg_cost, shares FROM portfolio WHERE ticker=?", (ticker,))
        row = c.fetchone()
        
        if trans_type == 'BUY':
            if row:
                old_cost, old_shares = row
                new_shares = old_shares + shares
                # 新成本 = (舊總值 + 本次買入總值) / 新股數
                new_cost = ((old_cost * old_shares) + total_cash_flow) / new_shares
                c.execute("UPDATE portfolio SET avg_cost=?, shares=? WHERE ticker=?", (new_cost, new_shares, ticker))
            else:
                new_cost = total_cash_flow / shares
                c.execute("INSERT INTO portfolio (ticker, avg_cost, shares, group_name) VALUES (?, ?, ?, ?)", 
                          (ticker, new_cost, shares, 'Default'))
        
        elif trans_type == 'SELL' and row:
            old_cost, old_shares = row
            if shares >= old_shares:
                c.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
            else:
                # 賣出時不影響剩餘庫存的平均成本，只扣股數
                c.execute("UPDATE portfolio SET shares=? WHERE ticker=?", (old_shares - shares, ticker))
                
        conn.commit()
        conn.close()
        return f"交易成功: {trans_type} {shares}股 {ticker} @ {price}"

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

# 初始化
DBManager.init_db()

# ==========================================
# 2. 量化運算引擎 (Quant Engines)
# ==========================================

class TechnicalEngine:
    """處理所有技術指標運算"""
    
    @staticmethod
    def calculate_all(df):
        if df.empty: return df
        df = df.copy()
        
        # 均線系統
        for ma in [5, 10, 20, 60, 120, 240]:
            df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
            
        # KD 指標 (Stochastic Oscillator)
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
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林通道 (Bollinger Bands)
        df['BB_Mid'] = df['MA20']
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Low'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['BB_Mid']
        
        # ATR (平均真實波幅) - 用於計算風險與停損
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df

class PatternEngine:
    """K線型態識別引擎 (Pattern Recognition)"""
    @staticmethod
    def detect_patterns(df):
        if len(df) < 5: return []
        signals = []
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. 均線排列
        if curr['MA5'] > curr['MA20'] > curr['MA60']: signals.append("🔥 多頭排列")
        if curr['MA5'] < curr['MA20'] < curr['MA60']: signals.append("❄️ 空頭排列")
        
        # 2. 黃金/死亡交叉
        if prev['K'] < prev['D'] and curr['K'] > curr['D'] and curr['K'] < 30: signals.append("📈 KD低檔金叉")
        if prev['MACD'] < prev['Signal'] and curr['MACD'] > curr['Signal']: signals.append("📈 MACD翻紅")
        
        # 3. K線型態 (簡化版)
        body = abs(curr['Close'] - curr['Open'])
        upper_shadow = curr['High'] - max(curr['Close'], curr['Open'])
        lower_shadow = min(curr['Close'], curr['Open']) - curr['Low']
        
        # 鎚子線 (底部反轉)
        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and curr['RSI'] < 40:
            signals.append("🔨 鎚子線 (疑似止跌)")
            
        # 長紅K (強勢)
        if (curr['Close'] - curr['Open']) / curr['Open'] > 0.03:
            signals.append("🕯️ 長紅K棒")
            
        # 布林突破
        if curr['Close'] > curr['BB_Up']: signals.append("🚀 突破布林上緣")
        if curr['BB_Width'] < 0.10: signals.append("⚡ 布林壓縮 (變盤前兆)")
        
        return signals

class RiskEngine:
    """風險計算與蒙地卡羅模擬 (使用 scipy)"""
    @staticmethod
    def calculate_sharpe(df, risk_free_rate=0.015):
        if len(df) < 30: return 0, 0
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) # 年化波動率
        cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1
        sharpe = (cagr - risk_free_rate) / volatility if volatility != 0 else 0
        return sharpe, volatility

    @staticmethod
    def monte_carlo_simulation(df, days=90, simulations=1000):
        """模擬未來股價走勢"""
        last_price = df['Close'].iloc[-1]
        returns = df['Close'].pct_change().dropna()
        daily_vol = returns.std()
        daily_drift = returns.mean() - (daily_vol ** 2) / 2
        
        # 生成隨機漫步路徑
        # 公式: Pt = Pt-1 * exp(drift + vol * Z)
        sim_paths = np.zeros((days, simulations))
        sim_paths[0] = last_price
        
        for t in range(1, days):
            random_shocks = norm.ppf(np.random.rand(simulations))
            sim_paths[t] = sim_paths[t-1] * np.exp(daily_drift + daily_vol * random_shocks)
            
        # 統計結果
        final_prices = sim_paths[-1]
        mean_final = np.mean(final_prices)
        p05 = np.percentile(final_prices, 5) # 95% 信賴區間下限 (VaR)
        p95 = np.percentile(final_prices, 95) # 95% 信賴區間上限
        win_rate = np.sum(final_prices > last_price) / simulations * 100
        
        return {
            "mean_price": mean_final,
            "var_price": p05,
            "upside_price": p95,
            "win_rate": win_rate,
            "paths": sim_paths[:, :50] # 只回傳前50條路徑供繪圖
        }

class ScoringEngine:
    """Willie AI 綜合評分系統 (0-100分)"""
    @staticmethod
    def calculate_score(d):
        score = 50 # 基礎分
        
        # 1. 趨勢面 (+- 20)
        if d['price'] > d['ma20']: score += 10
        if d['price'] > d['ma60']: score += 10
        if d['price'] < d['ma20']: score -= 10
        if d['price'] < d['ma60']: score -= 10
        
        # 2. 動能面 (+- 15)
        if d['k'] > d['d'] and d['k'] < 80: score += 5
        if d['rsi'] > 50: score += 5
        if d['macd'] > d['macd_sig']: score += 5
        
        # 3. 估值面 (+- 15)
        if d.get('pe'):
            if d['pe'] < 15: score += 15
            elif d['pe'] < 25: score += 5
            elif d['pe'] > 40: score -= 10
        
        # 4. 籌碼/波動面 (+- 10)
        if d.get('risk'):
            if d['risk']['sharpe'] > 1: score += 10
            if d['risk']['volatility'] > 0.5: score -= 5 # 波動太大扣分
            
        # 5. 殖利率 (+- 5)
        if d.get('yield', 0) > 4: score += 5
        
        return max(0, min(100, score))

# ==========================================
# 3. 數據抓取層 (Data Fetcher - Hybrid)
# ==========================================
class DataFetcher:
    @staticmethod
    def normalize_ticker(ticker):
        ticker = ticker.strip().upper()
        if ticker.isdigit(): ticker += ".TW"
        return ticker

    @staticmethod
    def fetch_simple(ticker):
        """輕量級抓取 (用於儀表板) - 避免卡頓"""
        ticker = DataFetcher.normalize_ticker(ticker)
        # 嘗試 Twstock
        if ticker[:2].isdigit():
            try:
                sid = ticker.replace(".TW", "")
                real = twstock.realtime.get(sid)
                if real['success']:
                    return {
                        "ticker": ticker,
                        "price": float(real['realtime']['latest_trade_price']),
                        "change_pct": 0.0, # 暫時忽略
                        "name": real['info']['name']
                    }
            except: pass
        
        # 嘗試 Yahoo
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                return {
                    "ticker": ticker, "price": curr, 
                    "change_pct": (curr-prev)/prev*100, "name": ticker
                }
        except: pass
        return None

    @staticmethod
    def fetch_full(ticker):
        """重量級抓取 (用於分析) - 含所有指標計算"""
        ticker = DataFetcher.normalize_ticker(ticker)
        cached = DBManager.get_cache(ticker, ttl_minutes=60)
        if cached: return cached
        
        data = {"ticker": ticker}
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y") # 抓2年以利回測與長均線
            if hist.empty: return None
            
            # 填補即時價
            if ticker[:2].isdigit():
                try:
                    real = twstock.realtime.get(ticker.replace(".TW", ""))
                    if real['success']:
                        data['price'] = float(real['realtime']['latest_trade_price'])
                        data['name'] = real['info']['name']
                except: pass
            
            if 'price' not in data: 
                data['price'] = hist['Close'].iloc[-1]
                data['name'] = stock.info.get('longName', ticker)

            # 執行運算引擎
            hist = TechnicalEngine.calculate_all(hist)
            sharpe, vol = RiskEngine.calculate_sharpe(hist)
            patterns = PatternEngine.detect_patterns(hist)
            
            # 基本面數據
            info = stock.info
            eps = info.get('trailingEps')
            pe = data['price'] / eps if eps and eps > 0 else None
            
            # 估值計算
            valuation = {}
            if eps:
                pe_s = hist['Close'] / eps
                valuation = {
                    "cheap": eps * pe_s.min(),
                    "fair": eps * pe_s.mean(),
                    "expensive": eps * pe_s.max()
                }

            # 蒙地卡羅模擬
            mc_sim = RiskEngine.monte_carlo_simulation(hist)
            
            # 整合數據包
            data.update({
                "change_pct": (data['price'] - hist['Close'].iloc[-2])/hist['Close'].iloc[-2]*100,
                "volume": hist['Volume'].iloc[-1],
                "pe": pe, "eps": eps, 
                "yield": info.get('dividendYield', 0) * 100,
                "market_cap": info.get('marketCap', 0),
                "sector": info.get('sector', 'Unknown'),
                "history_json": hist.reset_index().to_json(date_format='iso'),
                "valuation": valuation,
                "risk": {"sharpe": sharpe, "volatility": vol},
                "patterns": patterns,
                "monte_carlo": mc_sim,
                
                # 技術指標快照
                "ma20": hist['MA20'].iloc[-1], "ma60": hist['MA60'].iloc[-1],
                "k": hist['K'].iloc[-1], "d": hist['D'].iloc[-1],
                "rsi": hist['RSI'].iloc[-1],
                "macd": hist['MACD'].iloc[-1], "macd_sig": hist['Signal'].iloc[-1],
                "atr": hist['ATR'].iloc[-1]
            })
            
            # 最後計算總分
            data['score'] = ScoringEngine.calculate_score(data)
            
            DBManager.save_cache(ticker, data)
            return data
        except Exception as e:
            # print(f"Error {ticker}: {e}")
            return None

    @staticmethod
    def fetch_batch_simple(tickers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return list(filter(None, executor.map(DataFetcher.fetch_simple, tickers)))

    @staticmethod
    def fetch_batch_full(tickers):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            return list(filter(None, executor.map(DataFetcher.fetch_full, tickers)))

# ==========================================
# 4. 背景排程 (Automation)
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
# 5. UI 視覺化組件 (Components)
# ==========================================
def render_score_gauge(score):
    """繪製 AI 評分儀表"""
    color = "red" if score < 40 else "yellow" if score < 70 else "#00d4ff"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Willie AI Score", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "rgba(255,0,0,0.1)"},
                {'range': [40, 70], 'color': "rgba(255,255,0,0.1)"},
                {'range': [70, 100], 'color': "rgba(0,255,255,0.1)"}],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def render_radar_chart(d):
    """繪製六維雷達圖"""
    # 正規化數據到 0-100
    risk_score = max(0, 100 - d['risk']['volatility']*200)
    val_score = 80 if d.get('pe', 99) < 15 else 40
    tech_score = 80 if d['k'] > d['d'] else 40
    yield_score = min(100, d.get('yield', 0) * 15)
    trend_score = 80 if d['price'] > d['ma60'] else 30
    
    categories = ['估值', '技術', '籌碼(模擬)', '趨勢', '殖利率', '低風險']
    values = [val_score, tech_score, 60, trend_score, yield_score, risk_score]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name=d['name'],
        line=dict(color='#00d4ff'), fillcolor='rgba(0, 212, 255, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd")
    )
    st.plotly_chart(fig, use_container_width=True)

def render_monte_carlo(d):
    """繪製蒙地卡羅模擬路徑"""
    mc = d['monte_carlo']
    paths = mc['paths']
    
    fig = go.Figure()
    # 繪製前 30 條模擬路徑
    for i in range(min(30, paths.shape[1])):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(color='rgba(0, 212, 255, 0.1)', width=1), showlegend=False))
        
    # 繪製平均路徑
    mean_path = np.mean(paths, axis=1)
    fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='white', width=2, dash='dash'), name='平均預期'))
    
    fig.update_layout(
        title=f"未來 90 天股價模擬 (勝率: {mc['win_rate']:.1f}%)",
        yaxis_title="價格",
        height=350,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_pro_chart(d):
    """繪製專業級互動 K 線圖"""
    df = pd.read_json(d['history_json'])
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']); df.set_index('Date', inplace=True)
    elif 'index' in df.columns: df['index'] = pd.to_datetime(df['index']); df.set_index('index', inplace=True)
    
    # 建立多子圖
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{d['name']} 技術分析", "成交量 & MACD", "KD & RSI"))

    # Main: K線 + 均線 + 布林
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ffa726', width=1), name='MA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#29b6f6', width=1), name='MA60'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], fill='tonexty', fillcolor='rgba(255,255,255,0.05)', line=dict(width=0), name='BB'), row=1, col=1)
    
    # Sub 1: MACD + Volume
    colors = ['#ff5252' if r > 0 else '#69f0ae' for r in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='yellow', width=1), name='DIF'), row=2, col=1)
    
    # Sub 2: KD
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='#ffa726', width=1), name='K'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='#ab47bc', width=1), name='D'), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. 主程式 (Main Application)
# ==========================================

# --- 側邊欄 ---
with st.sidebar:
    st.title("🦅 Willie's Alpha")
    st.caption("V15.1 量化決策旗艦版")
    
    # 快速交易小工具
    with st.expander("⚡ 閃電下單 (Ledger)", expanded=True):
        col_t1, col_t2 = st.columns([2, 1])
        t_ticker = col_t1.text_input("代號", "2330").upper()
        t_action = col_t2.selectbox("動作", ["BUY", "SELL"])
        t_price = st.number_input("價格", 0.0, step=0.5)
        t_shares = st.number_input("股數", 1, step=1)
        if st.button("📝 寫入帳本"):
            msg = DBManager.record_transaction(DataFetcher.normalize_ticker(t_ticker), t_action, t_price, t_shares)
            st.success("已記錄！")
            time.sleep(1)
            st.rerun()
            
    # 全局更新
    if st.button("🔄 重整全站數據"):
        st.cache_data.clear()
        st.rerun()

# --- 主頁面 Tabs ---
tabs = st.tabs(["📊 戰情儀表板", "🔎 深度戰情室", "🎯 AI 選股雷達", "💰 資產帳本"])

# Tab 1: 儀表板 (Lite)
with tabs[0]:
    st.subheader("🌍 全球市場概況")
    # 定義要監控的商品
    items = {"^TWII":"加權指數", "^TWOII":"櫃買指數", "^SOX":"費半", "^IXIC":"那指", "GC=F":"黃金", "SI=F":"白銀", "CL=F":"原油", "USDTWD=X":"匯率"}
    
    # 快速抓取
    data_list = DataFetcher.fetch_batch_simple(list(items.keys()))
    
    # 渲染 Metrics
    cols = st.columns(4)
    for i, (k, v) in enumerate(items.items()):
        d = next((x for x in data_list if x['ticker'] == k), None)
        with cols[i % 4]:
            if d: 
                st.metric(v, f"{d['price']:,.2f}", f"{d['change_pct']:.2f}%")
            else: 
                st.metric(v, "N/A", "Loading...")
        if (i+1) % 4 == 0: st.write("")
    
    st.divider()
    
    # 庫存快速掃描 (如果有庫存)
    df_p = DBManager.get_portfolio()
    if not df_p.empty:
        st.subheader("💼 持股即時監控")
        # 這裡用 Full Fetch 因為要算分數
        if st.button("🚀 啟動庫存 AI 診斷"):
            with st.spinner("AI 正在為您的持股打分..."):
                p_data = DataFetcher.fetch_batch_full(df_p['ticker'].tolist())
            
            p_rows = []
            for d in p_data:
                p_rows.append({
                    "代號": d['ticker'], "名稱": d['name'], 
                    "AI評分": d['score'], "現價": d['price'], 
                    "漲跌%": d['change_pct'], "PE": d['pe'],
                    "趨勢": "🔥多頭" if d['price']>d['ma60'] else "❄️空頭"
                })
            
            st.dataframe(
                pd.DataFrame(p_rows).sort_values("AI評分", ascending=False),
                column_config={"AI評分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100)},
                use_container_width=True
            )
    else:
        st.info("尚無庫存，請至側邊欄新增交易。")

# Tab 2: 深度戰情室 (Deep Dive - The Core)
with tabs[1]:
    col1, col2 = st.columns([3, 1])
    search_input = col1.text_input("輸入代號進行全維度分析", "2330.TW").upper()
    if col2.button("開始分析"):
        DBManager.save_cache(DataFetcher.normalize_ticker(search_input), {}) # 強制刷新
        
    d = DataFetcher.fetch_full(search_input)
    
    if d:
        st.markdown(f"### {d['name']} ({d['ticker']})")
        
        # 1. 頂部核心數據矩陣
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("現價", d['price'], f"{d['change_pct']:.2f}%")
        m2.metric("本益比", f"{d['pe']:.1f}x" if d['pe'] else "-")
        m3.metric("殖利率", f"{d['yield']:.2f}%")
        m4.metric("夏普值 (Sharpe)", f"{d['risk']['sharpe']:.2f}")
        m5.metric("波動率 (Vol)", f"{d['risk']['volatility']*100:.1f}%")
        m6.metric("ATR (波動點數)", f"{d['atr']:.1f}")
        
        # 2. AI 分析面板
        col_ai_1, col_ai_2, col_ai_3 = st.columns([1, 1, 2])
        
        with col_ai_1:
            render_score_gauge(d['score'])
            
        with col_ai_2:
            render_radar_chart(d)
            
        with col_ai_3:
            st.markdown("#### 🧠 Willie AI 決策報告")
            
            # 生成文字分析
            signal_html = ""
            for p in d['patterns']:
                signal_html += f"<span style='background:#333;padding:4px 8px;border-radius:4px;margin-right:5px;border:1px solid #555'>{p}</span>"
            
            monte = d['monte_carlo']
            trend_str = "強勢多頭" if d['price'] > d['ma20'] and d['price'] > d['ma60'] else "空頭修正"
            
            st.markdown(f"""
            <div style="background:#1a1c24; padding:15px; border-radius:10px; border-left:4px solid var(--primary-color)">
                <p><strong>型態訊號：</strong> {signal_html if signal_html else '無明顯特殊型態'}</p>
                <p><strong>趨勢判斷：</strong> {trend_str} (股價與月季線關係)</p>
                <p><strong>風險預測：</strong> 蒙地卡羅模擬顯示，未來 90 天上漲機率為 <strong>{monte['win_rate']:.1f}%</strong>。</p>
                <p><strong>操作建議：</strong> AI 評分為 {d['score']} 分。
                   {'建議積極佈局 🚀' if d['score'] > 75 else '建議觀望或分批低接 🛡️' if d['score'] > 50 else '建議避開或減碼 ⚠️'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. 專業圖表區
        st.markdown("---")
        render_pro_chart(d)
        
        # 4. 估值與模擬區
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 💎 價值河流圖 (PE Band)")
            if d.get('valuation'):
                val = d['valuation']
                fig_val = go.Figure()
                fig_val.add_trace(go.Bar(y=['估值'], x=[val['cheap']], orientation='h', name='便宜', marker_color='#00fa9a', opacity=0.3))
                fig_val.add_trace(go.Bar(y=['估值'], x=[val['fair']-val['cheap']], base=val['cheap'], orientation='h', name='合理', marker_color='#29b6f6', opacity=0.3))
                fig_val.add_trace(go.Bar(y=['估值'], x=[val['expensive']-val['fair']], base=val['fair'], orientation='h', name='昂貴', marker_color='#ff4d4d', opacity=0.3))
                fig_val.add_trace(go.Scatter(y=['估值'], x=[d['price']], mode='markers+text', text=['現價'], marker=dict(size=15, color='white')))
                fig_val.update_layout(barmode='stack', height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_val, use_container_width=True)
            else: st.warning("無 EPS 數據")
            
        with c2:
            st.markdown("#### 🎲 蒙地卡羅未來模擬")
            render_monte_carlo(d)
            
    else:
        st.error("查無資料，請確認代號或稍後再試。")

# Tab 3: AI 選股雷達 (Screener)
with tabs[2]:
    st.subheader("🎯 智能選股雷達")
    st.markdown("透過多維度條件與 AI 評分，篩選出最強勢的標的。")
    
    with st.form("screener"):
        c1, c2, c3, c4 = st.columns(4)
        min_score = c1.slider("AI 評分 >", 0, 90, 60)
        max_pe = c2.slider("PE <", 10, 100, 25)
        min_yield = c3.slider("殖利率% >", 0.0, 10.0, 3.0)
        trend_only = c4.checkbox("僅限多頭排列 (價>月>季)", True)
        
        list_key = st.selectbox("掃描範圍", ["半導體 (Tech)", "金融 (Finance)", "航運 (Shipping)", "高股息 ETF", "全庫存"])
        
        btn = st.form_submit_button("🚀 啟動掃描")
        
    if btn:
        # 決定掃描名單
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        
        key_map = {
            "半導體 (Tech)": "list_tech", "金融 (Finance)": "list_finance",
            "航運 (Shipping)": "list_shipping", "高股息 ETF": "list_etf"
        }
        
        target_list = []
        if list_key == "全庫存":
            df_p = DBManager.get_portfolio()
            target_list = df_p['ticker'].tolist()
        else:
            cur.execute("SELECT value FROM system_config WHERE key=?", (key_map[list_key],))
            row = cur.fetchone()
            if row: target_list = row[0].split(",")
        conn.close()
        
        with st.spinner(f"AI 引擎正在分析 {len(target_list)} 檔標的..."):
            results = DataFetcher.fetch_batch_full(target_list)
            
        # 篩選邏輯
        filtered = []
        for r in results:
            keep = True
            if r['score'] < min_score: keep = False
            if r['pe'] and r['pe'] > max_pe: keep = False
            if r['yield'] < min_yield: keep = False
            if trend_only and not (r['price'] > r['ma20'] and r['price'] > r['ma60']): keep = False
            
            if keep: filtered.append(r)
            
        # 顯示結果
        if filtered:
            df_res = pd.DataFrame(filtered)
            df_display = df_res[['ticker', 'name', 'score', 'price', 'change_pct', 'pe', 'yield', 'volume']]
            st.success(f"篩選出 {len(filtered)} 檔優質標的！")
            st.dataframe(
                df_display.sort_values("score", ascending=False),
                column_config={
                    "score": st.column_config.ProgressColumn("AI評分", format="%d", min_value=0, max_value=100),
                    "change_pct": st.column_config.NumberColumn("漲跌%", format="%.2f%%"),
                    "yield": st.column_config.NumberColumn("殖利率%", format="%.2f%%")
                },
                use_container_width=True
            )
        else:
            st.warning("無符合條件的標的，請放寬篩選標準。")

# Tab 4: 資產帳本 (Ledger)
with tabs[3]:
    st.subheader("💰 資產損益與帳本")
    
    df_p = DBManager.get_portfolio()
    if not df_p.empty:
        # 抓取最新價
        tickers = df_p['ticker'].tolist()
        updates = DataFetcher.fetch_batch_simple(tickers)
        p_map = {u['ticker']: u['price'] for u in updates}
        
        rows = []
        tm, tc = 0, 0
        for _, r in df_p.iterrows():
            curr = p_map.get(r['ticker'], r['avg_cost'])
            mkt = curr * r['shares']
            cost = r['avg_cost'] * r['shares']
            tm += mkt; tc += cost
            rows.append({
                "代號": r['ticker'], "持有股數": r['shares'],
                "平均成本": r['avg_cost'], "現價": curr,
                "市值": int(mkt), "損益": int(mkt-cost),
                "報酬率%": (mkt-cost)/cost*100
            })
            
        m1, m2, m3 = st.columns(3)
        pnl = tm - tc
        m1.metric("總資產", f"${tm:,.0f}")
        m2.metric("總成本", f"${tc:,.0f}")
        m3.metric("未實現損益", f"${pnl:,.0f}", f"{pnl/tc*100:.2f}%")
        
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
        # 圓餅圖
        fig = px.pie(pd.DataFrame(rows), values='市值', names='代號', title='持倉分佈', hole=0.5)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("目前無庫存")
        
    st.markdown("### 📜 交易歷史")
    st.dataframe(DBManager.get_transactions(), use_container_width=True)
