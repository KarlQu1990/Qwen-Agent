from qwen_agent.agents.doc_qa import ParallelDocQA
from qwen_agent.gui import WebUI


def test():
    bot = ParallelDocQA(llm={"model": "qwen-plus", "generate_cfg": {"max_retries": 10}})
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "介绍实验方法"},
                {"file": "https://arxiv.org/pdf/2310.08560.pdf"},
            ],
        },
    ]
    for rsp in bot.run(messages):
        print("bot response:", rsp)


def app_gui():
    generate_cfg = {"max_tokens": 20000, "max_input_tokens": 20000}
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
        "generate_cfg": generate_cfg,
    }

    # Define the agent
    bot = ParallelDocQA(
        llm=llm_cfg,
        description="并行QA后用RAG召回内容并回答。支持文件类型：PDF/Word/PPT/TXT/HTML。使用与材料相同的语言提问会更好。",
    )

    chatbot_config = {
        "prompt.suggestions": [
            {"text": "合同中提到了几次甲方名称？这些甲方名称都一样吗？"},
            {"text": "合同中提到了几次乙方名称？这些乙方名称都一样吗？"},
            {
                "text": "合同的违约金赔偿利率是多少？合同签订日期是哪一年哪一月？搜索合同签订日期当年当月发布的中国人民银行授权全国银行间同业拆借中心公布的一年期贷款市场报价利率（LPR）是多少？计算合同的违约金赔偿利率大于 LPR 的 150%吗？"
            },
        ]
    }

    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == "__main__":
    # test()
    app_gui()
