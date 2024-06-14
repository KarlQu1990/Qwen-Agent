from qwen_agent.llm.bisheng import BishengHostLLM

cfg = {
    "host_base_url": "http://192.168.201.86:9221/v2.1/models",
    "model": "Qwen2-7B-Instruct",
    "model_type": "bisheng",
}

llm = BishengHostLLM(cfg)
for resp in llm.chat(
    [{"role": "user", "content": "你好"}],
    stream=False,
    extra_generate_cfg={"temperature": 0.9},
):
    print(resp)
