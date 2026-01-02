import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- 設定網頁配置 ---
st.set_page_config(page_title="Joymax 智慧選股助手", layout="wide", page_icon="📈")

# --- 內建觀察名單 ---
WATCH_LIST = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", 
    "2881.TW", "2882.TW", "2891.TW", "2002.TW", "1301.TW",
    "2382.TW", "2357.TW", "3231.TW", "2379.TW", "3008.TW",
    "2603.TW", "2609.TW", "2615.TW", "0050.TW", "0056.TW"
]

# --- 輔助函式：轉換 Dataframe 為 CSV ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# --- 輔助函式：批量掃描市場 ---
def scan_market(scan_type):
    data_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(WATCH_LIST)
    
    for i, ticker in enumerate(WATCH_LIST):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"正在掃描: {ticker} ({i+1}/{total})...")
        
        try:
            stock = yf.Ticker(ticker)
            # 抓取歷史數據
            hist = stock.history(period="1d")
            
            if hist.empty:
                continue

            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            # 嘗試抓取基本面數據
            # 雲端有時候 info 會抓取失敗，這裡做個備案
            try:
                info = stock.info
                eps = info.get('trailingEps') or info.get('forwardEps')
                name = info.get('longName', ticker)
            except:
                eps = None
                name = ticker
            
            # 計算本益比
            pe_ratio = current_price / eps if (eps and eps > 0) else 999
            
            data_list.append({
                "代號": ticker,
                "名稱": name,
                "股價": round(current_price, 2),
                "成交量": volume,
                "EPS": round(eps, 2) if eps else "N/A",
                "本益比": round(pe_ratio, 2) if isinstance(pe_ratio, float) and pe_ratio != 999 else "N/A"
            })
            
        except Exception as e:
            # 遇到錯誤就跳過，確保程式不崩潰
            print(f"Error fetching {ticker}: {e}")
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    # 【關鍵修正】如果全都抓不到，回傳一個空的但有欄位的 DataFrame，避免報錯
    if not data_list:
        return pd.DataFrame(columns=["代號", "名稱", "股價", "成交量", "EPS", "本益比"])
        
    return pd.DataFrame(data_list)

# --- 主程式介面 ---

st.title("📈 Joymax 智慧選股助手")
st.markdown("---")

# 建立側邊欄
with st.sidebar:
    st.header("🚀 快速掃描")
    st.caption("注意：雲端免費版有時會被阻擋，若掃描失敗請稍候再試。")
    
    if st.button("🔥 本日成交量 TOP 5"):
        st.session_state['scan_result'] = 'volume'
        
    if st.button("💎 低本益比 TOP 5"):
        st.session_state['scan_result'] = 'value'
        
    st.divider()
    
    st.header("🔍 個股深度分析")
    ticker_input = st.text_input("輸入股票代號", value="2330.TW").upper()
    lookback_period = st.selectbox("參考歷史區間", ["1y", "2y", "5y"], index=0)
    
    run_analysis = st.button("開始個股分析")

# --- 顯示掃描結果區塊 ---
if 'scan_result' in st.session_state:
    st.subheader("📊 市場掃描結果")
    
    df_result = scan_market(st.session_state['scan_result'])
    
    # 【關鍵修正】檢查是否有抓到資料
    if df_result.empty:
        st.warning("⚠️ 目前無法從雲端取得批量數據（可能被 Yahoo 暫時限流）。建議您使用左側的「個股深度分析」功能，通常仍可正常運作。")
    else:
        if st.session_state['scan_result'] == 'volume':
            st.caption("依「成交量」由大到小排序")
            final_df = df_result.sort_values(by="成交量", ascending=False).head(5)
            st.dataframe(final_df, use_container_width=True)
            
        elif st.session_state['scan_result'] == 'value':
            st.caption("依「本益比」由低到高排序 (排除虧損與 ETF)")
            # 確保本益比欄位不是字串 "N/A" 才能排序
            valid_pe = df_result[df_result["本益比"].apply(lambda x: isinstance(x, (int, float)))]
            if not valid_pe.empty:
                final_df = valid_pe.sort_values(by="本益比", ascending=True).head(5)
                st.dataframe(final_df, use_container_width=True)
            else:
                st.info("目前清單中沒有計算出有效本益比的股票。")
        
    del st.session_state['scan_result']
    st.divider()

# --- 個股深度分析邏輯 ---
if run_analysis or ticker_input:
    try:
        with st.spinner(f'正在深入分析 {ticker_input} ...'):
            stock = yf.Ticker(ticker_input)
            hist = stock.history(period=lookback_period)
            
            # 嘗試獲取 info，如果失敗則給預設值
            try:
                info = stock.info
            except:
                info = {}

            if hist.empty:
                st.error("找不到數據，請確認代號是否正確 (台股需加 .TW)。")
            else:
                # 數據計算
                current_price = hist['Close'].iloc[-1]
                eps = info.get('trailingEps') or info.get('forwardEps')
                
                if eps is None:
                    # 嘗試自己算 (如果沒有 info 但有歷史股價，雖然無法精準算 PE，但至少顯示股價)
                    st.warning("無法取得 EPS 數據，僅顯示股價走勢，無法計算目標價。")
                    st.metric("目前股價", f"{current_price:.2f}")
                    st.line_chart(hist['Close'])
                else:
                    implied_pe_series = hist['Close'] / eps
                    pe_min = implied_pe_series.min()
                    pe_mean = implied_pe_series.mean()
                    pe_max = implied_pe_series.max()
                    
                    target_cheap = eps * pe_min
                    target_fair = eps * pe_mean
                    target_expensive = eps * pe_max

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("目前股價", f"{current_price:.2f}")
                    col2.metric("EPS", f"{eps:.2f}")
                    col3.metric("目前本益比", f"{current_price/eps:.2f}")
                    
                    status = "⚪ 合理"
                    if current_price < target_cheap: status = "🟢 極度低估"
                    elif current_price < target_fair: status = "🔵 相對低估"
                    elif current_price > target_expensive: status = "🔴 過熱"
                    
                    col4.metric("評價", status)

                    valuation_data = {
                        "分析項目": ["樂觀目標價", "合理目標價", "保守目標價"],
                        "本益比倍數": [f"{pe_max:.2f}x", f"{pe_mean:.2f}x", f"{pe_min:.2f}x"],
                        "目標價格": [target_expensive, target_fair, target_cheap],
                        "潛在漲幅": [
                            (target_expensive - current_price) / current_price,
                            (target_fair - current_price) / current_price,
                            (target_cheap - current_price) / current_price
                        ]
                    }
                    df_val = pd.DataFrame(valuation_data)
                    
                    df_display = df_val.copy()
                    df_display["目標價格"] = df_display["目標價格"].map('{:,.2f}'.format)
                    df_display["潛在漲幅"] = df_display["潛在漲幅"].map('{:.2%}'.format)

                    st.table(df_display)

                    # 匯出按鈕
                    col_export_1, col_export_2 = st.columns([1, 4])
                    with col_export_1:
                        csv = convert_df(df_display)
                        st.download_button(
                            label="📥 下載報告",
                            data=csv,
                            file_name=f'{ticker_input}_valuation.csv',
                            mime='text/csv',
                        )
                    
                    # 繪圖
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[current_price], y=["位置"], mode='markers+text', marker=dict(size=20, color='black'), text=[f"現價 {current_price:.1f}"], textposition="top center", name='現價'))
                    fig.add_trace(go.Bar(x=[target_cheap], y=["位置"], orientation='h', marker=dict(color='green', opacity=0.3), name='低估'))
                    fig.add_trace(go.Bar(x=[target_fair-target_cheap], y=["位置"], base=target_cheap, orientation='h', marker=dict(color='blue', opacity=0.3), name='合理'))
                    fig.add_trace(go.Bar(x=[target_expensive-target_fair], y=["位置"], base=target_fair, orientation='h', marker=dict(color='red', opacity=0.3), name='昂貴'))
                    fig.update_layout(barmode='stack', height=200, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"發生錯誤: {e}")
