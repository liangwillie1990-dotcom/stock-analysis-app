import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import time

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 操盤手戰情室 V4", layout="wide", page_icon="📈")

# --- 資料庫：內建股票清單 ---
LIST_TOP_20 = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電", 
    "2303.TW": "聯電", "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", 
    "2002.TW": "中鋼", "1301.TW": "台塑", "2382.TW": "廣達", "2357.TW": "華碩", 
    "3231.TW": "緯創", "2379.TW": "瑞昱", "3008.TW": "大立光", "2603.TW": "長榮", 
    "2609.TW": "陽明", "2615.TW": "萬海", "0050.TW": "元大台灣50", "0056.TW": "元大高股息"
}

# 台灣 50 成分股 (示意，可視需要擴充)
LIST_TW50 = [
    "2330.TW", "2454.TW", "2317.TW", "2308.TW", "2303.TW", "2881.TW", "2882.TW", "2382.TW", "2891.TW", "2886.TW",
    "2412.TW", "3008.TW", "1301.TW", "2884.TW", "2892.TW", "2885.TW", "3034.TW", "3037.TW", "2357.TW", "2890.TW",
    "3231.TW", "3045.TW", "1303.TW", "2379.TW", "2880.TW", "2883.TW", "2887.TW", "5880.TW", "2912.TW", "2002.TW",
    "5871.TW", "2345.TW", "2395.TW", "4904.TW", "2327.TW", "3711.TW", "4938.TW", "1101.TW", "2408.TW", "2603.TW",
    "2801.TW", "6669.TW", "3017.TW", "2353.TW", "1326.TW", "2207.TW", "3035.TW", "5876.TW", "1216.TW", "2609.TW"
]

# --- 輔助函式：轉換 Dataframe 為 CSV ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 輔助函式：批量掃描 ---
def scan_market(ticker_list, scan_limit=None):
    data_list = []
    
    # 如果清單太長，為了避免當機，我們可以限制數量
    target_tickers = ticker_list[:scan_limit] if scan_limit else ticker_list
    
    progress_text = f"正在掃描 {len(target_tickers)} 檔股票 (每檔間隔 0.2 秒以防阻擋)..."
    my_bar = st.progress(0, text=progress_text)
    
    total = len(target_tickers)
    
    for i, ticker in enumerate(target_tickers):
        ticker = ticker.strip().upper()
        if not ticker: continue
        
        # 自動補上 .TW (如果使用者忘記打)
        if not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
            ticker += ".TW"

        my_bar.progress((i + 1) / total, text=f"正在分析 ({i+1}/{total}): {ticker} ...")
        
        try:
            time.sleep(0.2) # 防阻擋延遲
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                continue

            current_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_close
            volume = hist['Volume'].iloc[-1]
            pct_change = (current_close - prev_close) / prev_close
            
            # 抓取基本面
            eps = None
            try:
                info = stock.info
                eps = info.get('trailingEps') or info.get('forwardEps')
                name = info.get('longName', ticker)
            except:
                name = ticker

            # 本益比與目標價
            pe_ratio = 999
            target_str_fair = "N/A"
            target_str_cheap = "N/A"
            target_str_exp = "N/A"

            if eps and eps > 0:
                pe_ratio = current_close / eps
                # 簡單計算目標價
                implied_pe = hist['Close'] / eps
                t_cheap = eps * implied_pe.min()
                t_fair = eps * implied_pe.mean()
                t_exp = eps * implied_pe.max()
                
                target_str_cheap = f"{t_cheap:.1f}"
                target_str_fair = f"{t_fair:.1f}"
                target_str_exp = f"{t_exp:.1f}"
            
            # 針對 ETF 的處理
            if "00" in ticker[:2]:
                target_str_fair = "ETF"

            data_list.append({
                "代號": ticker,
                "名稱": name,
                "現價": round(current_close, 1),
                "漲跌幅%": round(pct_change * 100, 2),
                "成交量": volume,
                "本益比": round(pe_ratio, 1) if pe_ratio != 999 else "N/A",
                "保守價": target_str_cheap,
                "合理價": target_str_fair,
                "樂觀價": target_str_exp,
            })
            
        except Exception:
            continue
            
    my_bar.empty()
    return pd.DataFrame(data_list)

# --- 主介面 ---
st.title("📈 Joymax 操盤手戰情室 V4.0 (全台股擴充版)")
st.markdown("---")

# --- 側邊欄：掃描設定 ---
with st.sidebar:
    st.header("1. 設定掃描範圍")
    
    scan_source = st.radio(
        "選擇股票池來源：",
        ("🔥 精選 20 檔 (速度快)", "🏆 台灣 50 成分股 (約 30秒)", "📝 自訂/貼上清單")
    )
    
    target_list = []
    
    if scan_source == "🔥 精選 20 檔 (速度快)":
        target_list = list(LIST_TOP_20.keys())
        st.caption(f"掃描數量：{len(target_list)} 檔")
        
    elif scan_source == "🏆 台灣 50 成分股 (約 30秒)":
        target_list = LIST_TW50
        st.caption(f"掃描數量：{len(target_list)} 檔")
        
    elif scan_source == "📝 自訂/貼上清單":
        st.info("請輸入股票代號，用逗號或換行分隔 (例如：2330, 2317, 2603)")
        user_input = st.text_area("輸入代號區", "2330, 2317, 2603")
        # 處理使用者輸入
        if user_input:
            raw_list = user_input.replace("\n", ",").replace(" ", "").split(",")
            # 過濾空字串並補上 .TW (簡單防呆)
            target_list = [x for x in raw_list if x]
            st.caption(f"目前將掃描：{len(target_list)} 檔")

    st.divider()
    
    st.header("2. 執行快篩")
    # 按鈕區
    if st.button("🚀 開始掃描分析"):
        st.session_state['run_scan'] = True
        
    st.divider()
    
    st.header("3. 個股深度查詢")
    ticker_input = st.text_input("代號", value="2330.TW").upper()
    run_single = st.button("個股分析")

# --- 顯示掃描結果 ---
if st.session_state.get('run_scan'):
    st.subheader(f"📊 掃描結果：{scan_source}")
    
    if len(target_list) > 100:
        st.warning("⚠️ 您選擇的股票數量較多，請耐心等待 (預計每 10 檔需 3-5 秒)...")
    
    df_result = scan_market(target_list)
    
    if not df_result.empty:
        # 顯示互動表格
        st.dataframe(
            df_result, 
            use_container_width=True,
            column_config={
                "漲跌幅%": st.column_config.NumberColumn(
                    "漲跌幅%", format="%.2f %%"
                )
            }
        )
        
        # 快捷排序按鈕
        c1, c2, c3 = st.columns(3)
        if c1.button("按「成交量」排序"):
            st.dataframe(df_result.sort_values("成交量", ascending=False).head(10), use_container_width=True)
        if c2.button("按「本益比」排序 (找便宜)"):
            # 排除 N/A
            mask = df_result["本益比"].apply(lambda x: isinstance(x, (int, float)))
            st.dataframe(df_result[mask].sort_values("本益比").head(10), use_container_width=True)
        if c3.button("按「漲幅」排序 (找強勢)"):
            st.dataframe(df_result.sort_values("漲跌幅%", ascending=False).head(10), use_container_width=True)
            
        # 下載按鈕
        csv = convert_df(df_result)
        st.download_button("📥 下載完整 Excel/CSV", csv, "market_scan.csv", "text/csv")
    else:
        st.error("無法取得數據，請檢查代號格式 (台股需加 .TW) 或稍後再試。")

    # 執行完後重置，避免重複跑
    st.session_state['run_scan'] = False

# --- 個股分析 (保持原樣簡化版) ---
if run_single:
    st.divider()
    st.subheader(f"🔎 {ticker_input} 快速分析")
    try:
        stock = yf.Ticker(ticker_input)
        info = stock.info
        hist = stock.history(period="1y")
        curr = hist['Close'].iloc[-1]
        eps = info.get('trailingEps')
        
        c1, c2, c3 = st.columns(3)
        c1.metric("現價", f"{curr:.1f}")
        c2.metric("EPS", f"{eps}" if eps else "N/A")
        c3.metric("本益比", f"{curr/eps:.1f}" if eps else "N/A")
        
        st.line_chart(hist['Close'])
    except Exception as e:
        st.error(f"查無資料: {e}")
