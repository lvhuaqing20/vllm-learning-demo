from vllm import LLM, SamplingParams
import os

# 1. 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 配置模型路径 (你确认过的正确路径)
MODEL_PATH = "/home/leijianuo/.cache/huggingface/hub/input0"

print(f"正在加载模型 (限制显存以防崩溃)...")
# 【核心配置】融合了你的服务器防崩溃设置
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    max_model_len=2048,         # 限制上下文长度
    gpu_memory_utilization=0.4  # 只占 40% 显存，防止 OOM
)

# ---------------------------------------------------------
# 模式 A: 快思考 (System 1) - 直觉、快速、零废话
# ---------------------------------------------------------
def run_fast_mode(question):
    print(f"\n🚀 [快思考模式 System 1] 问题: {question}")
    
    # 温度设为 0，结果固定，不随机
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, stop=["<|im_end|>"])
    
    # 提示词：要求简短
    prompt = f"""<|im_start|>system
You are a concise assistant. Answer directly without explanation.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    output = llm.generate([prompt], sampling_params)[0]
    print(f"👉 结果: {output.outputs[0].text.strip()}")

# ---------------------------------------------------------
# 模式 B: 慢思考 (System 2) - 逻辑、推理、一步步来
# ---------------------------------------------------------
def run_slow_mode(question):
    print(f"\n🐢 [慢思考模式 System 2] 问题: {question}")
    
    # 温度设为 0.6，允许一点创造性思维
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024, stop=["<|im_end|>"])
    
    # 提示词：强制要求一步步思考 (Chain of Thought)
    prompt = f"""<|im_start|>system
You are a logical expert. You must think step by step before answering. Use <think> tags for your reasoning.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    output = llm.generate([prompt], sampling_params)[0]
    print(f"🧠 结果: {output.outputs[0].text.strip()}")

# ---------------------------------------------------------
# 测试主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    # 一个经典的逻辑陷阱题，快思考容易错，慢思考容易对
    question = "如果不考虑闰年，一年里有几个月有28天？"
    
    # 1. 运行快模式
    run_fast_mode(question)
    
    # 2. 运行慢模式
    run_slow_mode(question)
