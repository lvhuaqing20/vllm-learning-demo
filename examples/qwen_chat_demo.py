from vllm import LLM, SamplingParams
import os

# 1. 必选：设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 定义一个辅助函数，把普通问题包装成 Qwen 的对话格式
def get_prompt(question):
    return f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

# 准备问题（使用包装函数）
raw_questions = [
    "请你解释一下vllm框架",
    "法国的首都在哪里？",
]
# 这一步把 "你好" 变成了 "<|im_start|>user\n你好<|im_end|>..."
prompts = [get_prompt(q) for q in raw_questions]

# 3. 设置采样参数
# stop_token_ids 是为了让模型说完就停，不要自己在那自言自语
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.8, 
    max_tokens=512,
    stop=["<|im_end|>"] 
)

# 4. 加载模型
# 显存设为 0.4 (3090大约占9GB)，上下文限制在 2048 防止爆内存
actual_model_path = "/home/leijianuo/.cache/huggingface/hub/input0"

print(f"正在加载模型: {actual_model_path} ...")
llm = LLM(
    model=actual_model_path, 
    trust_remote_code=True, 
    max_model_len=2048,      # 【修改】降低上下文长度，防崩溃
    gpu_memory_utilization=0.4 # 【修改】给稍微多一点点空间，防止运行中OOM
)

# 5. 生成
outputs = llm.generate(prompts, sampling_params)

# 6. 打印结果
print("\n" + "=" * 50)
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    print(f"问: {raw_questions[i]}")
    print(f"答: {generated_text.strip()}")
    print("=" * 50)
