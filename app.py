import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 終極指揮官 V6", layout="wide", page_icon="🚀")

# ==========================================
# 1. 靜態資料定義 (名單與代號)
# ==========================================

# 國際與大盤指數
INDICES = {
    "^TWII": "加權指數", "^TWOII": "櫃買指數", 
    "^SOX": "費半指數", "^IXIC": "那斯達克"
}

# 產業龍頭 (用於熱力圖)
SECTORS = {
    "半導體": "2330.TW", "代工": "2317.TW", "IC設計": "2454.TW",
    "航運": "2603.TW", "金控": "2881.TW", "塑化": "1301.TW",
    "鋼鐵": "2002.TW", "AI伺服": "2382.TW", "重電": "1519.TW"
}

# 內建股票清單 (Top 20)
LIST_TOP_20 = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", 
    "2881.TW", "2882.TW", "2891.TW", "2002.TW", "1301.TW",
    "2382.TW", "2357.TW", "3231.TW", "2379.TW", "3008.TW",
    "2603.TW", "2609.TW", "2615.TW", "0050.TW", "0056.TW"
]

# 台灣 50 (簡化版)
LIST_TW50 = [
    "2330.TW", "2454.TW", "2317.TW", "2308.TW", "2303.TW", "2881.TW", "2882.TW", "2382.TW", "2891.TW", "2886.TW",
    "2412.TW", "3008.TW", "1301.TW", "2884.TW", "2892.TW", "2885.TW", "3034.TW", "3037.TW", "2357.TW", "2890.TW",
    "3231.TW", "3045.TW", "1303.TW", "2379.TW", "2880.TW", "2883.TW", "2887.TW", "5880.TW", "2912.TW", "2002.TW",
    "5871.TW", "2345.TW", "2395.TW", "4904.TW", "2327.TW", "3711.TW", "4938.TW", "1101.TW", "2408.TW", "2603.TW"
]

# ==========================================
# 2. 核心函式庫
# ==========================================

def get_simple_quote(ticker):
    """快速抓取單一報價 (給儀表板用)"""
    try:
        stock = yf.Ticker(ticker)
        # 抓 5 天是為了確保有上一個交易日資料
        hist = stock.history(period="5d")
        if hist.empty: return None
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        change = (current - prev) / prev * 100
        return current, change
    except:
        return None, None

def scan_market_detailed(ticker_list):
    """詳細掃描 (給快捷選單用，含 PE 計算與防阻擋)"""
    data = []
    # 進度條
    bar = st.progress(0, text="正在啟動雷達掃描...")
    total = len(ticker_list)
    
    for i, ticker in enumerate(ticker_list):
        ticker = ticker.strip().upper()
        if not ticker: continue
        if not ticker.endswith(".TW") and not ticker.endswith(".TWO"): ticker += ".TW"
        
        bar.progress((i+1)/total, text=f"分析中: {ticker} ({i+1}/{total})")
        
        try:
            time.sleep(0.2) # 關鍵延遲
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty: continue
            
            close = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else close
            volume = hist['Volume'].iloc[-1]
            pct = (close - prev) / prev
            
            # 52週高低
            high_52 = hist['High'].max()
            low_52 = hist['Low'].min()
            dist_high = (high_52 - close) / high_52
            dist_low = (close - low_52) / low_52

            # 基本面
            pe = 999
            try:
                info = stock.info
                eps = info.get('trailingEps') or info.get('forwardEps')
                name = info.get('longName', ticker)
            except:
                eps = None
                name = ticker
            
            # 目標價計算
            t_fair = "N/A"
            if eps and eps > 0:
                pe = close / eps
                pe_series = hist['Close'] / eps
                t_fair = f"{eps * pe_series.mean():.1f}"
            elif "00" in ticker[:2]:
                t_fair = "ETF"

            data.append({
                "代號": ticker, "名稱": name, "現價": round(close, 1),
                "漲跌%": round(pct*100, 2), "成交量": volume,
                "PE": round(pe, 1) if pe!=999 else "N/A",
                "合理價": t_fair,
                "_dist_high": dist_high, "_dist_low": dist_low # 排序用
            })
        except:
            continue
            
    bar.empty()
    return pd.DataFrame(data)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# ==========================================
# 3. 頁面佈局與邏輯
# ==========================================

# --- 側邊欄：您的快捷選單 (戰術區) ---
with st.sidebar:
    st.header("🎮 戰術控制台")
    
    # 1. 選擇彈藥庫 (股票來源)
    source = st.radio("股票池來源", ["Top 20 精選", "台灣 50", "自訂清單"])
    
    target_list = []
    if source == "Top 20 精選": target_list = LIST_TOP_20
    elif source == "台灣 50": target_list = LIST_TW50
    else:
        user_input = st.text_area("輸入代號 (逗號分隔)", "2330, 2603, 3035")
        if user_input:
            target_list = [x.strip() for x in user_input.replace("\n", ",").split(",") if x]

    st.divider()
    
    # 2. 快捷按鈕 (Trigger)
    st.subheader("🚀 一鍵掃描")
    btn_vol = st.button("🔥 成交爆量 Top 5")
    btn_pe = st.button("💎 低本益比 Top 5")
    btn_strong = st.button("📈 強勢股 Top 5")
    btn_weak = st.button("📉 弱勢股 Top 5")
    btn_near_high = st.button("☀️ 即將創高")
    btn_near_low = st.button("🌊 底部反彈")
    
    st.divider()
    
    # 3. 個股詳細
    st.subheader("🔍 單兵詳細分析")
    single_ticker = st.text_input("代號", "2330.TW").upper()
    btn_single = st.button("分析個股")

# --- 主畫面：上帝視角 (戰略區) ---
st.title("📊 Joymax 終極指揮官 V6")
st.caption("戰略看板 (Macro) + 戰術掃描 (Micro)")

# A. 大盤儀表板
cols = st.columns(4)
for i, (code, name) in enumerate(INDICES.items()):
    p, chg = get_simple_quote(code)
    with cols[i]:
        if p:
            st.metric(name, f"{p:,.0f}", f"{chg:.2f}%")
        else:
            st.metric(name, "連線中...")

st.markdown("---")

# B. 產業熱力圖 (保留您喜歡的圖表)
with st.expander("🏭 展開/收合：產業資金流向熱力圖", expanded=True):
    s_data = []
    for s_name, s_code in SECTORS.items():
        p, chg = get_simple_quote(s_code)
        if p:
            s_data.append({"產業": s_name, "漲跌幅": chg, "狀態": "紅" if chg>0 else "綠"})
    
    if s_data:
        df_sec = pd.DataFrame(s_data)
        fig = px.bar(df_sec, x='產業', y='漲跌幅', color='漲跌幅',
                     color_continuous_scale=['green', 'white', 'red'], range_color=[-3, 3],
                     title="各產業龍頭強弱指標", height=300)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# C. 掃描結果顯示區 (根據側邊欄按鈕觸發)
if 'scan_df' not in st.session_state:
    st.session_state['scan_df'] = None
if 'scan_title' not in st.session_state:
    st.session_state['scan_title'] = ""

# 邏輯處理：如果有按鈕被按下，執行掃描
scan_type = None
if btn_vol: scan_type = "vol"
elif btn_pe: scan_type = "pe"
elif btn_strong: scan_type = "strong"
elif btn_weak: scan_type = "weak"
elif btn_near_high: scan_type = "high"
elif btn_near_low: scan_type = "low"

if scan_type:
    st.session_state['scan_title'] = f"正在掃描：{source} ..."
    df_res = scan_market_detailed(target_list)
    st.session_state['scan_df'] = df_res
    st.session_state['scan_type'] = scan_type # 紀錄當下類型以利排序

# 顯示掃描結果
if st.session_state['scan_df'] is not None and not st.session_state['scan_df'].empty:
    df = st.session_state['scan_df']
    sType = st.session_state.get('scan_type')
    
    st.subheader(f"🎯 掃描結果報告 ({source})")
    
    final_df = df.copy()
    if sType == "vol":
        st.caption("依成交量排序")
        final_df = df.sort_values("成交量", ascending=False).head(5)
    elif sType == "pe":
        st.caption("依本益比排序 (排除虧損)")
        mask = df["PE"].apply(lambda x: isinstance(x, (int, float)))
        final_df = df[mask].sort_values("PE").head(5)
    elif sType == "strong":
        st.caption("依漲幅排序")
        final_df = df.sort_values("漲跌%", ascending=False).head(5)
    elif sType == "weak":
        st.caption("依跌幅排序")
        final_df = df.sort_values("漲跌%", ascending=True).head(5)
    elif sType == "high":
        st.caption("離 52 週新高最近 (準備突破)")
        final_df = df.sort_values("_dist_high").head(5)
    elif sType == "low":
        st.caption("離 52 週新低最近 (超跌)")
        final_df = df.sort_values("_dist_low").head(5)

    # 顯示表格 (隱藏內部計算欄位)
    show_cols = ["代號", "名稱", "現價", "漲跌%", "成交量", "PE", "合理價"]
    st.dataframe(final_df[show_cols], use_container_width=True)
    
    # 下載按鈕
    csv = convert_df(final_df[show_cols])
    st.download_button("📥 下載此清單", csv, "scan_result.csv", "text/csv")

# D. 個股單獨分析 (保留最受歡迎的 PE Band)
if btn_single:
    st.markdown("---")
    st.subheader(f"🔎 {single_ticker} 深度分析")
    try:
        stock = yf.Ticker(single_ticker)
        hist = stock.history(period="1y")
        info = stock.info
        eps = info.get('trailingEps') or info.get('forwardEps')
        
        if not hist.empty and eps:
            curr = hist['Close'].iloc[-1]
            pe_series = hist['Close'] / eps
            p_min, p_mean, p_max = pe_series.min(), pe_series.mean(), pe_series.max()
            t_cheap, t_fair, t_exp = eps*p_min, eps*p_mean, eps*p_max
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{curr:.1f}")
            c2.metric("EPS", f"{eps:.2f}")
            c3.metric("本益比", f"{curr/eps:.1f}x")
            
            status = "⚪ 合理"
            if curr < t_cheap: status = "🟢 低估"
            elif curr > t_exp: status = "🔴 過熱"
            c4.metric("評價", status)
            
            # 視覺化 PE Band
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[curr], y=[0], mode='markers+text', text=["現價"], marker=dict(size=15, color='black')))
            fig.add_trace(go.Bar(x=[t_cheap], y=[0], orientation='h', name='便宜', marker_color='green', opacity=0.3))
            fig.add_trace(go.Bar(x=[t_fair-t_cheap], y=[0], base=t_cheap, orientation='h', name='合理', marker_color='blue', opacity=0.3))
            fig.add_trace(go.Bar(x=[t_exp-t_fair], y=[0], base=t_fair, orientation='h', name='昂貴', marker_color='red', opacity=0.3))
            fig.update_layout(height=200, barmode='stack', yaxis=dict(showticklabels=False), margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.line_chart(hist['Close'])
        else:
            st.error("無法取得完整數據 (可能缺 EPS)")
            
    except Exception as e:
        st.error(f"查詢失敗: {e}")
