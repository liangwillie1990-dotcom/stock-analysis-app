import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 台股總覽戰情室 V5", layout="wide", page_icon="📊")

# --- 核心數據定義 ---

# 1. 國際與大盤指數
INDICES = {
    "^TWII": "🇹🇼 加權指數 (大盤)",
    "^TWOII": "🇹🇼 櫃買指數 (中小型)",
    "^SOX": "🇺🇸 費半指數 (半導體)",
    "^IXIC": "🇺🇸那斯達克 (科技)",
    "^GSPC": "🇺🇸 S&P 500",
}

# 2. 產業代表性龍頭 (用龍頭股漲跌代表該產業資金流向)
SECTORS = {
    "半導體": "2330.TW",   # 台積電
    "代工組裝": "2317.TW", # 鴻海
    "IC設計": "2454.TW",   # 聯發科
    "航運": "2603.TW",     # 長榮
    "金融": "2881.TW",     # 富邦金
    "塑化": "1301.TW",     # 台塑
    "鋼鐵": "2002.TW",     # 中鋼
    "AI伺服器": "2382.TW", # 廣達
    "重電綠能": "1519.TW", # 華城
    "營建": "2501.TW",     # 國建
}

# --- 輔助函式 ---
def get_stock_data(ticker):
    """快速抓取單一股票/指數的最新數據與均線"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo") # 抓半年以計算均線
        
        if hist.empty: return None
        
        close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_pct = (close - prev_close) / prev_close * 100
        
        # 計算均線 (月線20MA, 季線60MA)
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        ma60 = hist['Close'].rolling(60).mean().iloc[-1]
        
        # 判斷多空趨勢
        trend = "盤整"
        if close > ma20 and close > ma60: trend = "🔥 強多格局"
        elif close < ma20 and close < ma60: trend = "❄️ 空頭弱勢"
        elif close > ma20: trend = "📈 短多支撐"
        elif close < ma20: trend = "📉 短線轉弱"

        return {
            "price": close,
            "change": change_pct,
            "ma20": ma20,
            "trend": trend
        }
    except:
        return None

# --- 主程式 ---

st.title("📊 Joymax 台股總覽戰情室 V5.0")
st.caption("由上而下 (Top-Down) 觀察：國際股市 -> 台股大盤 -> 產業流向")
st.markdown("---")

# ==========================================
# 區塊 1: 國際與大盤儀表板 (Macro View)
# ==========================================
st.subheader("1. 🌍 全球關鍵指數 (多空風向球)")

# 建立 5 個欄位顯示指數
cols = st.columns(5)

# 為了效能，我們一次性顯示，不使用進度條
for i, (ticker, name) in enumerate(INDICES.items()):
    data = get_stock_data(ticker)
    with cols[i]:
        if data:
            color = "normal"
            if data['change'] > 0: color = "off" # Streamlit metric 綠色代表漲需要反過來設定? 不，預設紅漲綠跌需用 delta_color
            
            st.metric(
                label=name,
                value=f"{data['price']:,.0f}",
                delta=f"{data['change']:.2f}%",
            )
            st.caption(f"趨勢: {data['trend']}")
        else:
            st.metric(label=name, value="N/A")

st.info("💡 觀察重點：費半指數 (^SOX) 通常領先連動台股；櫃買指數 (^TWOII) 代表內資與中小型股活躍度。")
st.markdown("---")

# ==========================================
# 區塊 2: 產業資金流向 (Sector Heatmap)
# ==========================================
st.subheader("2. 🏭 產業資金流向 (類股龍頭漲跌)")

# 掃描產業龍頭
sector_data = []
for sector_name, ticker in SECTORS.items():
    data = get_stock_data(ticker)
    if data:
        sector_data.append({
            "產業": sector_name,
            "龍頭股": ticker,
            "漲跌幅%": data['change'],
            "狀態": "上漲" if data['change'] > 0 else "下跌"
        })

if sector_data:
    df_sector = pd.DataFrame(sector_data)
    
    # 使用 Plotly 畫出漂亮的長條圖
    fig = px.bar(
        df_sector, 
        x='產業', 
        y='漲跌幅%', 
        color='漲跌幅%',
        color_continuous_scale=['green', 'white', 'red'], # 綠跌紅漲
        range_color=[-3, 3], # 設定顏色區間 -3% 到 +3%
        title="今日各產業強弱勢一覽 (紅強綠弱)",
        text_auto='.2f'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 文字快評
    top_sector = df_sector.loc[df_sector['漲跌幅%'].idxmax()]
    low_sector = df_sector.loc[df_sector['漲跌幅%'].idxmin()]
    st.success(f"🔥 今日最強族群：**{top_sector['產業']}** (漲幅 {top_sector['漲跌幅%']:.2f}%)")
    st.error(f"❄️ 今日最弱族群：**{low_sector['產業']}** (漲幅 {low_sector['漲跌幅%']:.2f}%)")

st.markdown("---")

# ==========================================
# 區塊 3: 個股詳細查詢 (保留 V4 功能)
# ==========================================
st.subheader("3. 🔍 個股深度分析")

col1, col2 = st.columns([1, 3])
with col1:
    ticker_input = st.text_input("輸入個股代號", value="2330.TW").upper()
    if st.button("開始分析"):
        st.session_state['run_stock'] = True

with col2:
    if st.session_state.get('run_stock'):
        try:
            stock = yf.Ticker(ticker_input)
            hist = stock.history(period="1y")
            info = stock.info
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                eps = info.get('trailingEps') or info.get('forwardEps')
                
                # 簡單計算目標價
                if eps:
                    pe = current / eps
                    pe_band = hist['Close'] / eps
                    target_fair = eps * pe_band.mean()
                    upside = (target_fair - current) / current
                    
                    st.write(f"**{ticker_input} 分析結果**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("現價", f"{current:.1f}")
                    c2.metric("本益比", f"{pe:.1f}x")
                    c3.metric("合理目標價", f"{target_fair:.1f}", delta=f"{upside:.2%}")
                    
                    st.line_chart(hist['Close'])
                else:
                    st.warning("無 EPS 數據，僅顯示股價。")
                    st.line_chart(hist['Close'])
        except Exception as e:
            st.error(f"查詢失敗: {e}")
