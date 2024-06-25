from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI


def test():
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
    }
    bot = Assistant(llm=llm_cfg)
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "介绍图一"},
                {"file": "https://arxiv.org/pdf/1706.03762.pdf"},
            ],
        }
    ]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
    }

    # Define the agent
    bot = Assistant(
        llm=llm_cfg,
        name="Assistant",
        description="使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。",
    )
    chatbot_config = {
        "prompt.suggestions": [
            {"text": "介绍图一"},
            {"text": "第二章第一句话是什么？"},
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == "__main__":
    # test()
    app_gui()
