import streamlit as st
import re
from vllm import LLM, SamplingParams
import os

# -----------------------------------------------------------------------------
# 1. 基础配置区
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Qwen-0.5B 双模式演示", layout="wide")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 你的模型绝对路径 (请确认这个路径是正确的)
MODEL_PATH = "/home/leijianuo/.cache/huggingface/hub/input0"

# -----------------------------------------------------------------------------
# 2. 核心功能函数
# -----------------------------------------------------------------------------

# 回调函数：切换模式时清空聊天记录，防止模型“学坏”
def reset_chat():
    st.session_state.messages = []

@st.cache_resource
def load_model():
    print(f"正在从 {MODEL_PATH} 加载模型...")
    # 显存防崩溃配置
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=8192,         # 【修改】脑容量扩充到 4096
        gpu_memory_utilization=0.6  # 【修改】只占 40% 显存，防止 OOM
    )
    return llm

# 尝试加载模型
try:
    with st.spinner("正在唤醒 Qwen 模型，请稍候..."):
        llm = load_model()
    st.success("模型加载就绪！")
except Exception as e:
    st.error(f"模型加载失败，请检查路径: {e}")

# -----------------------------------------------------------------------------
# 3. 侧边栏 (UI)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 思考模式")
    
    # 模式选择 (绑定了 on_change=reset_chat，一换模式就清屏)
    mode = st.radio(
        "选择 System:",
        ("快思考 (System 1)", "慢思考 (System 2)"),
        captions=["直觉反应，速度快", "深度推理，逻辑强"],
        on_change=reset_chat 
    )
    
    st.markdown("---")
    st.markdown("### 💡 测试建议")
    st.markdown("- **快模式**: 9.11 和 9.9 哪个大？")
    st.markdown("- **慢模式**: Strawberry 有几个 r？")
    
    st.markdown("---")
    # 手动清空按钮
    if st.button("🗑️ 清空对话"):
        reset_chat()
        st.rerun()

# -----------------------------------------------------------------------------
# 4. 主聊天界面
# -----------------------------------------------------------------------------
st.title("🤖 Qwen-0.5B 快慢思考双模式")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# 5. 处理用户输入与生成
# -----------------------------------------------------------------------------
if prompt := st.chat_input("请输入你的问题..."):
    # A. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. 生成回答
    with st.chat_message("assistant"):
        # 根据模式设定 System Prompt 和 参数
        if mode == "快思考 (System 1)":
            # 快模式：温度0，强制简短
            temp = 0.0
            max_tk = 5000  # 【修改】增加到 512，防止话没说完
            sys_prompt = "You are a concise assistant. Answer the user's question directly. Do NOT use <think> tags. Do not explain your reasoning."
        else:
            # 慢模式：温度0.6，强制 CoT (Chain of Thought)
            temp = 0.6
            max_tk = 7000 # 【修改】增加到 3500，允许长篇大论
            sys_prompt = "You are a logical expert. You must think step by step before answering. Use <think> tags for your reasoning."

        # 构造 ChatML 格式的 Prompt
        # 注意：为了演示效果清晰，这里仅发送当前单轮对话，避免历史记录干扰 System Prompt 的效果
        full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temp, 
            top_p=0.8, 
            max_tokens=max_tk, 
            stop=["<|im_end|>"]
        )

        # 调用 vLLM 生成 (耗时操作)
        with st.spinner(f'正在使用 {mode} 思考中...'):
            outputs = llm.generate([full_prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
# 【新增】暴力清洗逻辑：如果是快模式，强制删除 <think> 标签及其内容
        if mode == "快思考 (System 1)":
            # 这里的正则意思是：找到 <think> 和 </think> 之间的所有内容，替换为空
            # flags=re.DOTALL 表示允许匹配换行符
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # 显示并保存回答
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
