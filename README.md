# 🧠 Qwen-0.5B 快慢思考双模式演示 (Fast/Slow Thinking Demo)

这是一个基于 **vLLM** 和 **Streamlit** 构建的大模型演示应用，专为显存受限的服务器环境 (如 RTX 3090) 设计。它展示了如何通过 Prompt Engineering 和正则清洗逻辑，让同一个模型表现出两种截然不同的思考模式。

## ✨ 项目亮点

- **⚡️ 快思考 (System 1)**: 
  - 模拟直觉反应，温度设为 0。
  - **自动清洗**: 内置 Python 正则表达式 (`re`) 强制过滤 `<think>` 标签，仅输出最终结论，解决小模型无法抑制思考过程的问题。
  - 适用场景：简单常识、直接问答。

- **🐢 慢思考 (System 2)**: 
  - 模拟深度推理，温度设为 0.6。
  - **思维链展示**: 强制保留并展示 `<think>` 标签，完整呈现推理过程 (Chain of Thought)。
  - 适用场景：逻辑陷阱题 (如 "9.11 和 9.9 谁大")、数学计算。

- **💾 显存深度优化**: 
  - 针对多用户共享的学校服务器环境优化。
  - 显存占用限制为 **40%** (`gpu_memory_utilization=0.4`)，防止 OOM。
  - 上下文长度扩容至 **4096** token。

## 🛠️ 环境依赖

- Python 3.10+
- CUDA 11.8
- vLLM
- Streamlit

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install vllm streamlit
2. 启动 Web 界面
Bash

streamlit run examples/web_demo.py --server.port 8501 --server.address 0.0.0.0
3. 本地访问 (SSH Tunnel)
如果服务器没有公网 IP，请在本地电脑终端运行以下命令进行端口映射：

Bash

# 请替换 your_username, server_ip 和 ssh_port
ssh -L 8501:localhost:8501 your_username@server_ip -p ssh_port
然后打开浏览器访问：http://localhost:8501
