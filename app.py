"""
Joymax Zen V22.0 - Pure Valuation
Author: Gemini AI
Description: The simplest tool for PE-based valuation. No distractions.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from fake_useragent import UserAgent

# ==========================================
# 0. 設定與偽裝
# ==========================================
st.set_page_config(page_title="Joymax 估價單", page_icon="🧮")

def get_session():
    """建立偽裝連線，避免 Yahoo 阻擋"""
    session = requests.Session()
    try:
        ua = UserAgent()
        session.headers['User-Agent'] = ua.random
    except:
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    return session

# ==========================================
# 1. 核心估價邏輯
# ==========================================
def calculate_valuation(ticker):
    # 格式處理
    ticker = ticker.strip().upper()
    if ticker.isdigit(): ticker += ".TW"
    
    try:
        session = get_session()
        stock = yf.Ticker(ticker, session=session)
        
        # 1. 抓取歷史股價 (用來算過去本益比區間)
        hist = stock.history(period="5y") # 抓5年數據較準確
        if hist.empty:
            hist = stock.history(period="1y") # 至少抓1年
            
        if hist.empty: return None, "無法取得歷史股價"

        # 2. 抓取 EPS (每股盈餘)
        info = stock.info
        eps = info.get('trailingEps') or info.get('forwardEps')
        
        # 如果 Yahoo info 抓不到，嘗試自己算 (近四季 EPS 總和)
        # 這邊為了極簡，如果真的沒有 EPS 就報錯
        if not eps:
            return None, f"無法取得 {ticker} 的 EPS 數據，無法估價。"

        # 3. 計算本益比 (PE) 歷程
        # 公式：當天股價 / 目前 EPS (簡化計算，假設 EPS 穩定)
        # 更嚴謹的做法是用當下的 TTM EPS，但在簡易版中我們用最新 EPS 估算區間
        pe_series = hist['Close'] / eps
        
        # 去除極端值 (例如虧損時的負值，或異常高值)
        pe_series = pe_series[pe_series > 0] 
        pe_series = pe_series[pe_series < 100] # 去除 PE > 100 的異常狀況
        
        if pe_series.empty:
            return None, "本益比數據異常 (可能是虧損公司)，無法估值。"

        # 4. 算出位階
        pe_min = pe_series.min()      # 最低本益比
        pe_mean = pe_series.mean()    # 平均本益比
        pe_max = pe_series.max()      # 最高本益比
        current_price = hist['Close'].iloc[-1]
        current_pe = current_price / eps

        result = {
            "name": info.get('longName', ticker),
            "ticker": ticker,
            "current_price": current_price,
            "current_pe": current_pe,
            "eps": eps,
            "valuation": {
                "cheap": eps * pe_min,      # 便宜價
                "fair": eps * pe_mean,      # 合理價
                "expensive": eps * pe_max   # 昂貴價
            },
            "pe_stats": {
                "min": pe_min,
                "mean": pe_mean,
                "max": pe_max
            }
        }
        return result, None

    except Exception as e:
        return None, f"發生錯誤: {str(e)}"

# ==========================================
# 2. 極簡 UI
# ==========================================
st.title("🧮 Joymax 極簡估價")
st.caption("輸入代號 → 取得 低估 / 合理 / 樂觀 價格")

col_in, col_btn = st.columns([3, 1])
ticker_input = col_in.text_input("股票代號", "2330")
btn_run = col_btn.button("開始估價", type="primary")

if btn_run or ticker_input:
    if not ticker_input:
        st.warning("請輸入代號")
    else:
        with st.spinner(f"正在計算 {ticker_input} 的本益比河流..."):
            data, error = calculate_valuation(ticker_input)
            
            if error:
                st.error(error)
            else:
                val = data['valuation']
                curr = data['current_price']
                
                # --- 顯示結果 ---
                st.header(f"{data['name']} ({data['ticker']})")
                
                # 三大指標
                c1, c2, c3 = st.columns(3)
                c1.metric("目前股價", f"{curr:.1f}")
                c2.metric("EPS (每股盈餘)", f"{data['eps']:.2f} 元")
                c3.metric("目前本益比", f"{data['current_pe']:.1f} 倍")
                
                st.divider()
                
                # 估價結果 (重點)
                st.subheader("🎯 目標價位分析")
                
                v1, v2, v3 = st.columns(3)
                
                # 判斷目前位置的顏色
                def get_color(price, target):
                    return "normal"
                
                v1.metric("🟢 低估 (便宜價)", f"{val['cheap']:.1f}", f"{val['cheap'] - curr:.1f}", delta_color="normal")
                v2.metric("🔵 合理 (平均價)", f"{val['fair']:.1f}", f"{val['fair'] - curr:.1f}", delta_color="normal")
                v3.metric("🔴 樂觀 (昂貴價)", f"{val['expensive']:.1f}", f"{val['expensive'] - curr:.1f}", delta_color="normal")
                
                # 視覺化位階條 (簡單直觀)
                fig = go.Figure()
                
                # 背景區間
                fig.add_trace(go.Bar(
                    y=['位階'], x=[val['cheap']], orientation='h', 
                    name='低估區', marker_color='#4caf50', opacity=0.6,
                    hovertemplate='低估區: 0 ~ %{x:.1f}'
                ))
                fig.add_trace(go.Bar(
                    y=['位階'], x=[val['fair'] - val['cheap']], orientation='h', 
                    name='合理區', marker_color='#2196f3', opacity=0.6, base=val['cheap'],
                    hovertemplate='合理區'
                ))
                fig.add_trace(go.Bar(
                    y=['位階'], x=[val['expensive'] - val['fair']], orientation='h', 
                    name='昂貴區', marker_color='#f44336', opacity=0.6, base=val['fair'],
                    hovertemplate='昂貴區'
                ))
                
                # 當前價格標記
                fig.add_trace(go.Scatter(
                    y=['位階'], x=[curr], mode='markers+text', 
                    marker=dict(symbol='diamond', size=20, color='black', line=dict(width=2, color='white')),
                    text=[f"現價 {curr}"], textposition="top center",
                    name='目前股價'
                ))
                
                fig.update_layout(
                    barmode='stack', 
                    height=200, 
                    xaxis=dict(title='股價', range=[val['cheap']*0.8, val['expensive']*1.1]),
                    yaxis=dict(showticklabels=False),
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 詳細數據說明
                with st.expander("查看計算細節"):
                    st.write(f"計算依據：過去 5 年本益比區間")
                    st.write(f"- 最低本益比: {data['pe_stats']['min']:.1f} 倍")
                    st.write(f"- 平均本益比: {data['pe_stats']['mean']:.1f} 倍")
                    st.write(f"- 最高本益比: {data['pe_stats']['max']:.1f} 倍")
                    st.write(f"- 使用 EPS: {data['eps']:.2f}")
