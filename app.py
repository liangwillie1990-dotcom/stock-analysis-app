import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 操盤手戰情室", layout="wide", page_icon="📈")

# --- 內建觀察名單 (字典格式：代號 -> 中文名稱) ---
# 您可以在此自由新增股票
WATCH_LIST = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", 
    "2308.TW": "台達電", "2303.TW": "聯電", "2881.TW": "富邦金", 
    "2882.TW": "國泰金", "2891.TW": "中信金", "2002.TW": "中鋼", 
    "1301.TW": "台塑", "2382.TW": "廣達", "2357.TW": "華碩", 
    "3231.TW": "緯創", "2379.TW": "瑞昱", "3008.TW": "大立光",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", 
    "0050.TW": "元大台灣50", "0056.TW": "元大高股息"
}

# --- 輔助函式：轉換 Dataframe 為 CSV ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 輔助函式：批量掃描市場 ---
def scan_market(scan_type):
    data_list = []
    # 建立進度條
    progress_text = "正在掃描市場數據，請稍候..."
    my_bar = st.progress(0, text=progress_text)
    
    total = len(WATCH_LIST)
    tickers = list(WATCH_LIST.keys())
    
    for i, ticker in enumerate(tickers):
        # 更新進度
        my_bar.progress((i + 1) / total, text=f"正在分析: {WATCH_LIST[ticker]} ({ticker})...")
        
        try:
            stock = yf.Ticker(ticker)
            # 抓取 1 年數據以計算 52 週高低與本益比區間
            hist = stock.history(period="1y")
            
            if hist.empty:
                continue

            # --- 基礎數據 ---
            current_close = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_close
            open_price = hist['Open'].iloc[-1]
            high_price = hist['High'].iloc[-1]
            low_price = hist['Low'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            # --- 技術指標計算 ---
            pct_change = (current_close - prev_close) / prev_close  # 漲跌幅
            day_range = (high_price - low_price) / prev_close       # 當日振幅
            
            year_high = hist['High'].max()
            year_low = hist['Low'].min()
            
            # 距離 52 週高低點的百分比
            dist_to_high = (year_high - current_close) / year_high
            dist_to_low = (current_close - year_low) / year_low

            # --- 估值數據 (目標價計算) ---
            # 嘗試抓取 EPS
            try:
                info = stock.info
                eps = info.get('trailingEps') or info.get('forwardEps')
            except:
                eps = None
            
            # 初始化目標價字串
            target_str_cheap = "N/A"
            target_str_fair = "N/A"
            target_str_exp = "N/A"
            pe_ratio = 999

            if eps and eps > 0:
                pe_ratio = current_close / eps
                
                # 計算歷史本益比區間
                implied_pe_series = hist['Close'] / eps
                pe_min = implied_pe_series.min()
                pe_mean = implied_pe_series.mean()
                pe_max = implied_pe_series.max()
                
                # 計算目標價
                t_cheap = eps * pe_min
                t_fair = eps * pe_mean
                t_exp = eps * pe_max
                
                # 格式化顯示
                target_str_cheap = f"{t_cheap:.1f}"
                target_str_fair = f"{t_fair:.1f}"
                target_str_exp = f"{t_exp:.1f}"

            # 彙整資料
            data_list.append({
                "代號": ticker,
                "名稱": WATCH_LIST[ticker],
                "現價": round(current_close, 1),
                "漲跌幅%": round(pct_change * 100, 2),
                "成交量": volume,
                "本益比": round(pe_ratio, 1) if pe_ratio != 999 else "N/A",
                "保守價(低估)": target_str_cheap,
                "合理價(平均)": target_str_fair,
                "樂觀價(昂貴)": target_str_exp,
                # 隱藏欄位用於排序
                "_day_range": day_range,
                "_dist_to_high": dist_to_high,
                "_dist_to_low": dist_to_low
            })
            
        except Exception as e:
            continue
            
    my_bar.empty()
    
    if not data_list:
        return pd.DataFrame()
        
    return pd.DataFrame(data_list)

# --- 主程式介面 ---

st.title("📈 Joymax 操盤手戰情室 V3.0")
st.markdown("---")

# 建立側邊欄
with st.sidebar:
    st.header("⚡ 分析師快篩")
    st.info("針對內建 20 檔權值股進行掃描")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔥 成交爆量"): st.session_state['scan'] = 'volume'
        if st.button("📈 強勢排行"): st.session_state['scan'] = 'gainer'
        if st.button("☀️ 即將創高"): st.session_state['scan'] = 'near_high'
        if st.button("🎢 波動劇烈"): st.session_state['scan'] = 'volatile'
        
    with col_btn2:
        if st.button("💎 低本益比"): st.session_state['scan'] = 'value'
        if st.button("📉 弱勢排行"): st.session_state['scan'] = 'loser'
        if st.button("🌊 底部反彈"): st.session_state['scan'] = 'near_low'

    st.divider()
    
    st.header("🔍 個股深度分析")
    ticker_input = st.text_input("輸入代號", value="2330.TW").upper()
    
    # 新增更多時間區間選項
    period_map = {
        "1個月 (短線)": "1mo",
        "3個月 (季線)": "3mo",
        "6個月 (半年線)": "6mo",
        "1年 (年線)": "1y",
        "2年 (長期)": "2y",
        "5年 (超長期)": "5y"
    }
    selected_label = st.selectbox("參考歷史區間", list(period_map.keys()), index=3)
    lookback_period = period_map[selected_label]
    
    run_analysis = st.button("開始分析")

# --- 顯示掃描結果 ---
if 'scan' in st.session_state:
    st.subheader("📊 市場掃描儀表板")
    
    df = scan_market(st.session_state['scan'])
    
    if df.empty:
        st.warning("⚠️ 無法取得數據，請稍後再試。")
    else:
        # 根據按鈕類型進行排序與篩選
        scan_type = st.session_state['scan']
        final_df = df.copy()
        
        if scan_type == 'volume':
            st.caption("篩選條件：成交量最大 Top 5")
            final_df = df.sort_values(by="成交量", ascending=False).head(5)
            
        elif scan_type == 'value':
            st.caption("篩選條件：本益比最低 Top 5 (排除虧損)")
            # 過濾掉 N/A
            mask = final_df["本益比"].apply(lambda x: isinstance(x, (int, float)))
            final_df = final_df[mask].sort_values(by="本益比", ascending=True).head(5)
            
        elif scan_type == 'gainer':
            st.caption("篩選條件：今日漲幅最高 Top 5")
            final_df = df.sort_values(by="漲跌幅%", ascending=False).head(5)
            
        elif scan_type == 'loser':
            st.caption("篩選條件：今日跌幅最重 Top 5")
            final_df = df.sort_values(by="漲跌幅%", ascending=True).head(5)
            
        elif scan_type == 'near_high':
            st.caption("篩選條件：距離 52 週高點最近 (準備突破)")
            final_df = df.sort_values(by="_dist_to_high", ascending=True).head(5)
            
        elif scan_type == 'near_low':
            st.caption("篩選條件：距離 52 週低點最近 (超跌觀察)")
            final_df = df.sort_values(by="_dist_to_low", ascending=True).head(5)
            
        elif scan_type == 'volatile':
            st.caption("篩選條件：當日高低震盪幅度最大")
            final_df = df.sort_values(by="_day_range", ascending=False).head(5)

        # 移除內部運算用的隱藏欄位，只顯示給使用者看的
        display_cols = ["代號", "名稱", "現價", "漲跌幅%", "本益比", "保守價(低估)", "合理價(平均)", "樂觀價(昂貴)"]
        st.dataframe(final_df[display_cols], use_container_width=True)
        
    del st.session_state['scan']
    st.divider()

# --- 個股深度分析 (邏輯不變，僅配合新選項) ---
if run_analysis or ticker_input:
    try:
        with st.spinner(f'正在深入分析 {ticker_input} ...'):
            stock = yf.Ticker(ticker_input)
            # 使用使用者選擇的時間區間
            hist = stock.history(period=lookback_period)
            
            try: info = stock.info
            except: info = {}

            if hist.empty:
                st.error(f"找不到 {ticker_input} 的數據。")
            else:
                current_price = hist['Close'].iloc[-1]
                # 取得中文名稱 (如果在清單內) 或是英文原名
                stock_name = WATCH_LIST.get(ticker_input, info.get('longName', ticker_input))
                
                st.subheader(f"📌 {stock_name} ({ticker_input})")

                eps = info.get('trailingEps') or info.get('forwardEps')
                
                if eps is None:
                    st.warning("無 EPS 數據，僅顯示股價走勢。")
                    st.line_chart(hist['Close'])
                else:
                    # 計算邏輯
                    pe_series = hist['Close'] / eps
                    pe_min = pe_series.min()
                    pe_mean = pe_series.mean()
                    pe_max = pe_series.max()
                    
                    target_cheap = eps * pe_min
                    target_fair = eps * pe_mean
                    target_expensive = eps * pe_max

                    # 頂部指標
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("目前股價", f"{current_price:.2f}")
                    col2.metric("EPS", f"{eps:.2f}")
                    col3.metric("本益比 (PE)", f"{current_price/eps:.2f}")
                    
                    status = "⚪ 合理"
                    if current_price < target_cheap: status = "🟢 低估 (便宜)"
                    elif current_price < target_fair: status = "🔵 相對低"
                    elif current_price > target_expensive: status = "🔴 過熱 (昂貴)"
                    col4.metric("評價", status)

                    # 詳細表格
                    df_val = pd.DataFrame({
                        "分析項目": ["樂觀 (昂貴)", "平均 (合理)", "保守 (低估)"],
                        "PE 倍數": [f"{pe_max:.2f}x", f"{pe_mean:.2f}x", f"{pe_min:.2f}x"],
                        "目標價格": [target_expensive, target_fair, target_cheap],
                        "潛在漲幅": [
                            (target_expensive - current_price) / current_price,
                            (target_fair - current_price) / current_price,
                            (target_cheap - current_price) / current_price
                        ]
                    })
                    
                    # 格式化
                    df_show = df_val.copy()
                    df_show["目標價格"] = df_show["目標價格"].map('{:,.2f}'.format)
                    df_show["潛在漲幅"] = df_show["潛在漲幅"].map('{:.2%}'.format)
                    st.table(df_show)

                    # 匯出報告
                    csv = convert_df(df_show)
                    st.download_button(
                        label="📥 下載分析報告 (CSV)",
                        data=csv,
                        file_name=f'{ticker_input}_report.csv',
                        mime='text/csv',
                    )
                    
                    # 繪製位階圖
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[current_price], y=["位置"], mode='markers+text', marker=dict(size=20, color='black'), text=[f"現價 {current_price:.1f}"], textposition="top center", name='現價'))
                    fig.add_trace(go.Bar(x=[target_cheap], y=["位置"], orientation='h', marker=dict(color='green', opacity=0.3), name='低估'))
                    fig.add_trace(go.Bar(x=[target_fair-target_cheap], y=["位置"], base=target_cheap, orientation='h', marker=dict(color='blue', opacity=0.3), name='合理'))
                    fig.add_trace(go.Bar(x=[target_expensive-target_fair], y=["位置"], base=target_fair, orientation='h', marker=dict(color='red', opacity=0.3), name='昂貴'))
                    fig.update_layout(barmode='stack', height=200, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="價格")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 歷史走勢圖
                    st.subheader(f"📉 歷史股價 ({selected_label})")
                    st.line_chart(hist['Close'])

    except Exception as e:
        st.error(f"發生錯誤: {e}")
