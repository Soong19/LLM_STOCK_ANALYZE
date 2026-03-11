import os
import argparse
import sys
import logging
from datetime import datetime

import akshare as ak
import pandas as pd
import pandas_ta as ta
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_stock_data(symbol):
    logging.info(f"Fetching data for symbol: {symbol}")
    # Try A-share
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        if not df.empty:
            logging.info("Successfully fetched A-share data.")
            return df, "A_SHARE"
    except Exception as e:
        logging.debug(f"A-share fetch failed: {e}")
        
    # Try HK-share
    try:
        df = ak.stock_hk_hist(symbol=symbol, period="daily", adjust="qfq")
        if not df.empty:
            logging.info("Successfully fetched HK-share data.")
            return df, "HK_SHARE"
    except Exception as e:
        logging.debug(f"HK-share fetch failed: {e}")
        
    raise ValueError(f"Could not fetch data for {symbol}. Ensure it's a valid A-share (e.g., 600519) or HK-share (e.g., 00700) code.")

def calculate_indicators(df):
    logging.info("Calculating technical indicators...")
    
    close_col = "收盘"
    if close_col not in df.columns:
        # Fallback if names are different
        logging.warning(f"Column '{close_col}' not found. Available columns: {df.columns.tolist()}")
        if "close" in df.columns:
            close_col = "close"
    
    # Sort just in case
    if "日期" in df.columns:
        df = df.sort_values(by="日期").reset_index(drop=True)
    elif "date" in df.columns:
        df = df.sort_values(by="date").reset_index(drop=True)
    
    # Ensure numeric types
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
    
    # Moving Averages
    df['MA5'] = ta.sma(df[close_col], length=5)
    df['MA10'] = ta.sma(df[close_col], length=10)
    df['MA20'] = ta.sma(df[close_col], length=20)
    
    # MACD
    macd = ta.macd(df[close_col])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        
    # RSI
    df['RSI'] = ta.rsi(df[close_col], length=14)
    
    # BOLL
    boll = ta.bbands(df[close_col], length=20)
    if boll is not None:
        df = pd.concat([df, boll], axis=1)
        
    return df

def get_basic_info(symbol, market):
    info = {}
    try:
        if market == "A_SHARE":
            prefix = "sh" if symbol.startswith("6") else "sz" if symbol.startswith("0") or symbol.startswith("3") else "bj"
            # Some symbols might not be available in lg indicators, ignore errors silently
            df = ak.stock_a_indicator_lg(symbol=prefix + symbol)
            if not df.empty:
                latest = df.iloc[-1]
                info['pe'] = latest.get('pe', 'N/A')
                info['pb'] = latest.get('pb', 'N/A')
                info['dv_ratio'] = latest.get('dv_ratio', 'N/A')
                info['total_mv'] = latest.get('total_mv', 'N/A')
    except Exception as e:
        logging.debug(f"Could not fetch basic metrics: {e}")
    return info

def generate_prompt(symbol, df, info):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    date_val = latest.get('日期', latest.get('date', '未知'))
    close_val = latest.get('收盘', latest.get('close', '未知'))
    pct_val = latest.get('涨跌幅', '未知')
    vol_val = latest.get('成交量', latest.get('volume', '未知'))
    turnover_val = latest.get('换手率', '未知')

    # Formatting helper
    def fmt(val):
        if pd.isna(val) or val == 'N/A':
            return '未知'
        try:
            return f"{float(val):.2f}"
        except:
            return val

    prompt = f"""
请你作为一名资深的股票分析师，对股票代码：{symbol} 进行全方位的技术面和基本面分析，并给出评分。

【最新日线交易数据】
- 日期：{date_val}
- 收盘价：{close_val}
- 涨跌幅：{pct_val}%
- 成交量：{vol_val}
- 换手率：{turnover_val}%

【技术指标】
- MA5: {fmt(latest.get('MA5'))}
- MA10: {fmt(latest.get('MA10'))}
- MA20: {fmt(latest.get('MA20'))}
- MACD_12_26_9: {fmt(latest.get('MACD_12_26_9'))} (前一日: {fmt(prev.get('MACD_12_26_9'))})
- MACDh_12_26_9 (MACD柱): {fmt(latest.get('MACDh_12_26_9'))}
- MACDs_12_26_9 (DEA/信号线): {fmt(latest.get('MACDs_12_26_9'))}
- RSI_14: {fmt(latest.get('RSI_14'))}
- BOLL (中轨 BBM_20_2.0_2.0): {fmt(latest.get('BBM_20_2.0_2.0'))}
- BOLL (上轨 BBU_20_2.0_2.0): {fmt(latest.get('BBU_20_2.0_2.0'))}
- BOLL (下轨 BBL_20_2.0_2.0): {fmt(latest.get('BBL_20_2.0_2.0'))}

【基本面及估值（如有）】
- 市盈率(PE): {info.get('pe', '未知')}
- 市净率(PB): {info.get('pb', '未知')}
- 股息率(DV Ratio): {info.get('dv_ratio', '未知')}
- 总市值(万元): {info.get('total_mv', '未知')}

请结合以上数据进行深度分析，包括：
1. **趋势分析**：根据均线系统分析当前多空趋势。
2. **动能与震荡**：结合MACD和RSI指标分析买卖动能和超买超卖情况。
3. **支撑与压力**：根据BOLL轨道分析当前的支撑位与压力位。
4. **量价配合**：分析成交量变化和换手率。
5. **价值分析**：根据PE、PB和股息率简评其估值水平（如适用）。
6. **综合打分**：在0-10分之间进行严格打分（0分为极度看跌，10分为极度看涨）。
7. **操作建议**：给出明确的建议（如：买入、持有、减仓、卖出、观望等）及核心理由。

请将输出结果使用Markdown格式进行美化输出，结构清晰。将【综合打分】与【操作建议】高亮显示。
"""
    return prompt

def analyze_with_llm(prompt):
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
    
    if not api_key:
        logging.warning("LLM_API_KEY environment variable is not set. Skipping LLM call and printing prompt instead.")
        return f"*** LLM_API_KEY not set ***\n\nPrompt to be sent:\n{prompt}"
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    logging.info(f"Calling LLM API using model: {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一位专业的A股和港股股票分析师。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="AI Stock Analyst")
    parser.add_argument("symbol", help="Stock code (e.g., 600519 or 00700)")
    args = parser.parse_args()
    
    symbol = args.symbol
    
    try:
        # 1. Fetch data
        df, market = get_stock_data(symbol)
        
        # 2. Add indicators
        df = calculate_indicators(df)
        
        # 3. Get basic info
        info = get_basic_info(symbol, market)
        
        # 4. Generate prompt
        prompt = generate_prompt(symbol, df, info)
        logging.info("Data fetched and indicators calculated successfully.")
        
        # 5. Call LLM
        analysis = analyze_with_llm(prompt)
        
        # 6. Output specifically formatted for GitHub Actions Step Summary
        print("\n" + "="*50 + "\n")
        print(analysis)
        print("\n" + "="*50 + "\n")
        
        step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary_file:
            with open(step_summary_file, "a", encoding="utf-8") as f:
                f.write(f"# AI Stock Analysis for {symbol}\n\n")
                f.write(analysis)
                f.write("\n\n---\n*Auto-generated by AI Stock Analysis Action*")
                
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
