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

def generate_single_stock_data_text(symbol, df, info):
    if df is None or df.empty:
        return f"### 股票代码：{symbol}\n【获取数据失败或无数据】\n"
        
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

    text = f"""
### 股票代码：{symbol}

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
"""
    return text

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
    parser.add_argument("symbol", help="Stock codes separated by commas (e.g., 600519,00700)")
    args = parser.parse_args()
    
    symbol_input = args.symbol
    symbols = [s.strip() for s in symbol_input.split(',') if s.strip()]
    if len(symbols) > 20:
        logging.warning("最多只支持同时分析20个股票。已截断前20个。")
        symbols = symbols[:20]
        
    if not symbols:
        logging.error("没有提供有效的股票代码。")
        sys.exit(1)
        
    all_data_texts = []
    
    for symbol in symbols:
        try:
            # 1. Fetch data
            df, market = get_stock_data(symbol)
            
            # 2. Add indicators
            df = calculate_indicators(df)
            
            # 3. Get basic info
            info = get_basic_info(symbol, market)
            
            # 4. Generate data text
            data_text = generate_single_stock_data_text(symbol, df, info)
            all_data_texts.append(data_text)
            logging.info(f"Data for {symbol} processed successfully.")
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            all_data_texts.append(f"### 股票代码：{symbol}\n【获取数据失败：{e}】\n")
            
    combined_data = "\n---\n".join(all_data_texts)
    
    prompt = f"""
请你作为一名资深的股票分析师，对以下 {len(symbols)} 只股票（代码：{', '.join(symbols)}）进行全方位的技术面和基本面综合分析，并输出一份综合报告。

以下是各只股票的最新数据：
{combined_data}

请结合以上数据进行深度分析。报告需包含以下两部分：

第一部分：个股分析（对于成功获取数据的股票）
对于每只股票，请逐一分析并在标题处务必显示**股票代码以及对应的公司名称**：
1. **当前状态与趋势**：结合均线及BOLL分析当前的趋势及支撑压力位。
2. **动能与量价**：结合MACD、RSI及成交量换手率分析动能情况。
3. **估值简评**：结合PE/PB及股息率等简评估值水平（若有）。
4. **综合打分与建议**：给出严格的打分（0-10分，0分为极度看跌，10分为极度看涨）。**操作建议必须且只能从以下五个级别中选择一个：“重仓买入”、“轻仓买入”、“观望”、“部分卖出”、“全部卖出”**。给出核心操作理由。此外，如果你的建议是买入或卖出（包含重仓买入、轻仓买入、部分卖出、全部卖出四种），**请务必给出明确的建议操作价格（或价格区间）**。

第二部分：所有股票概括总结
请在报告结尾，对所有分析的股票进行一个**简要的整体概括总结**，简短概述即可，**不需要进行横向对比**。

请将输出结果使用Markdown格式进行美化输出，结构清晰。将【公司名称】、【综合打分】与【操作建议（及建议价格）】高亮显示。
"""
    
    try:
        # 5. Call LLM
        analysis = analyze_with_llm(prompt)
        
        # 6. Output specifically formatted for GitHub Actions Step Summary
        print("\n" + "="*50 + "\n")
        print(analysis)
        print("\n" + "="*50 + "\n")
        
        step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary_file:
            with open(step_summary_file, "a", encoding="utf-8") as f:
                f.write(f"# AI Stock Analysis Report\n\n")
                f.write(f"**Analyzed Symbols:** {', '.join(symbols)}\n\n")
                f.write(analysis)
                f.write("\n\n---\n*Auto-generated by AI Stock Analysis Action*")
                
    except Exception as e:
        logging.error(f"Error calling LLM: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
