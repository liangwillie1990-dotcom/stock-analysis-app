import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 設定網頁配置
st.set_page_config(page_title="個股本益比估價模型", layout="wide")

# 標題與說明
st.title("📈 個股本益比 (P/E) 估價分析 App")
st.markdown("""
輸入股票代號（台股請加上 `.TW`，例如 `2330.TW`），系統將根據**過去一年的本益比區間**來推算合理價格。
""")

# 側邊欄輸入
with st.sidebar:
    st.header("查詢設定")
    ticker_symbol = st.text_input("輸入股票代號", value="2330.TW").upper()
    lookback_period = st.selectbox("參考歷史區間", ["1y", "2y", "5y"], index=0)
    st.caption("註：台股請加 .TW (上市) 或 .TWO (上櫃)")

    if st.button("開始分析"):
        should_run = True
    else:
        should_run = False

# 主程式邏輯
if should_run or ticker_symbol:
    try:
        with st.spinner(f'正在分析 {ticker_symbol} 的數據...'):
            # 1. 獲取股票數據
            stock = yf.Ticker(ticker_symbol)
            
            # 獲取歷史股價
            hist = stock.history(period=lookback_period)
            
            if hist.empty:
                st.error("找不到該股票數據，請檢查代號是否正確。")
                st.stop()

            # 2. 獲取關鍵財務數據
            # 嘗試取得不同欄位的 EPS，以防資料缺漏
            info = stock.info
            eps = info.get('trailingEps') or info.get('forwardEps')
            
            current_price = hist['Close'].iloc[-1]
            
            # 如果真的抓不到 EPS
            if eps is None:
                st.warning("無法取得該股票的 EPS 數據，無法進行本益比分析。")
                st.stop()

            # 3. 計算歷史本益比區間 (PE Band)
            implied_pe_series = hist['Close'] / eps
            
            pe_min = implied_pe_series.min()
            pe_mean = implied_pe_series.mean()
            pe_max = implied_pe_series.max()
            current_pe = current_price / eps

            # 4. 計算目標價格
            target_cheap = eps * pe_min
            target_fair = eps * pe_mean
            target_expensive = eps * pe_max

            # --- 顯示結果區域 ---
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("目前股價", f"{current_price:.2f}")
            col2.metric("每股盈餘 (EPS)", f"{eps:.2f}")
            col3.metric("目前本益比", f"{current_pe:.2f} 倍")
            
            # 判斷目前狀態
            status = ""
            if current_price < target_cheap:
                status = "🟢 極度低估"
            elif current_price < target_fair:
                status = "🔵 相對低估"
            elif current_price > target_expensive:
                status = "🔴 過熱/樂觀"
            else:
                status = "⚪ 合理區間"
            
            col4.metric("評價狀態", status)

            st.divider()

            # 估值分析表格
            st.subheader("📊 本益比估價結果")
            
            valuation_data = {
                "情境": ["樂觀 (昂貴)", "平均 (合理)", "保守 (低估)"],
                "參考本益比倍數": [f"{pe_max:.2f} x", f"{pe_mean:.2f} x", f"{pe_min:.2f} x"],
                "目標價格": [f"{target_expensive:.2f}", f"{target_fair:.2f}", f"{target_cheap:.2f}"],
                "潛在漲跌幅": [
                    f"{((target_expensive - current_price) / current_price * 100):.2f}%",
                    f"{((target_fair - current_price) / current_price * 100):.2f}%",
                    f"{((target_cheap - current_price) / current_price * 100):.2f}%"
                ]
            }
            st.table(pd.DataFrame(valuation_data))

            # 視覺化圖表
            st.subheader("🎯 股價位階圖")
            
            fig = go.Figure()

            # 添加主要股價線
            fig.add_trace(go.Scatter(
                x=[current_price], y=["股價位置"],
                mode='markers+text',
                marker=dict(size=20, color='black'),
                text=[f"目前: {current_price:.1f}"],
                textposition="top center",
                name='目前股價'
            ))

            # 添加區間棒狀圖
            fig.add_trace(go.Bar(
                x=[target_cheap], y=["股價位置"],
                orientation='h',
                marker=dict(color='green', opacity=0.3),
                name='低估區間'
            ))
            
            fig.add_trace(go.Bar(
                x=[target_fair - target_cheap], y=["股價位置"],
                base=target_cheap,
                orientation='h',
                marker=dict(color='blue', opacity=0.3),
                name='合理區間'
            ))
            
            fig.add_trace(go.Bar(
                x=[target_expensive - target_fair], y=["股價位置"],
                base=target_fair,
                orientation='h',
                marker=dict(color='red', opacity=0.3),
                name='樂觀區間'
            ))

            fig.update_layout(
                xaxis_title="股價",
                barmode='stack',
                height=250,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"📉 {ticker_symbol} 過去 {lookback_period} 股價走勢")
            st.line_chart(hist['Close'])

    except Exception as e:
        st.error(f"發生錯誤: {e}")
        st.info("常見原因：輸入了錯誤的代號，或該股票沒有足夠的財務數據。")