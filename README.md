# AI Stock Analysis Tool (AI股票分析助手)

这是一个自动化的 AI 股票分析工具，支持A股和港股。它通过 `akshare` 获取最新的日线交易数据与财务指标，利用 `pandas-ta` 计算关键技术指标（如 MA、MACD、RSI、BOLL等），最终将所有数据整理并发送给您配置的大语言模型（LLM），生成兼顾基本面与技术面的综合评分（0-10分）和操作建议。

**最核心的特点是**：部署在 GitHub Actions 上，可以在手机上安装 [GitHub App](https://github.com/mobile) 随时随地输入股票代码，一键触发分析，并在运行日志的 `Summary`（摘要）中直接查看精美的 Markdown 分析报告！

## 🚀 部署与使用指南

### 1. Fork 本仓库
点击右上角的 `Fork` 按钮，将本仓库复制到你的个人账号下。

### 2. 配置大模型 API 密钥 (Secrets)
在你的仓库中，点击 `Settings` -> `Secrets and variables` -> `Actions`，点击 `New repository secret`，添加以下三个环境变量：

* **`LLM_API_KEY`** (必填): 你的大模型 API Key（例如 OpenAI 提供的 `sk-xxxx`，或者是 DeepSeek、通义千问等兼容 OpenAI 接口的 Key）。
* **`LLM_BASE_URL`** (可选，默认 `https://api.openai.com/v1`): 如果你使用的是其他模型或国内代理中转，将这里设置为它的接口地址（例如 `https://api.deepseek.com/v1` 或 `https://dashscope.aliyuncs.com/compatible-mode/v1`）。
* **`LLM_MODEL`** (可选，默认 `gpt-3.5-turbo`): 你希望调用的模型名称（例如 `gpt-4o`, `deepseek-chat`, `qwen-plus` 等）。

### 3. 如何在手机上触发？
1. 下载并登录手机端的 **GitHub App**。
2. 进入你 Fork 的这个仓库。
3. 点击 **Actions** 选项卡。
4. 选择左侧的 **AI Stock Analysis** 工作流。
5. 点击 **Run workflow** 面板，输入你想要分析的股票代码（如 `600519` 代表贵州茅台，`00700` 代表腾讯控股）。
6. 运行结束后，点击进入具体的 Run，在 **Summary**（摘要）界面滑动即可查看排版精良的 AI 分析报告。

## 🛠️ 本地运行

如果你希望在本地电脑上运行：

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量 (Mac/Linux)
export LLM_API_KEY="sk-xxxx"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"

# 运行脚本
python main.py 600519
```

## 📊 支持的股票类型
* **A股**: 上海(`6`开头), 深圳(`0`或`3`开头), 北交所(`4`或`8`开头)。直接输入数字代码，如 `600519` 或 `000001`。
* **港股**: 通常是5位数字代码，如 `00700` (腾讯), `03690` (美团)。

## ⚠️ 免责声明
本工具生成的投资建议与打分 **仅供参考**，不构成任何实质性的投资建议及操作依据。股市有风险，投资需谨慎！所引用的金融数据全部来自开源API，可能存在延迟或误差。
