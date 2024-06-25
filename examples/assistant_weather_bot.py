"""A weather forecast assistant implemented by assistant"""

import os
from typing import Optional

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")
os.environ["AMAP_TOKEN"] = ""  # your key


def init_agent_service():
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
    }
    system = (
        "你扮演一个天气预报助手，你具有查询天气和画图能力。"
        "你需要查询相应地区的天气，然后调用给你的画图工具绘制一张城市的图，并从给定的诗词文档中选一首相关的诗词来描述天气，不要说文档以外的诗词。"
    )

    tools = ["image_gen", "amap_weather"]
    bot = Assistant(
        llm=llm_cfg,
        name="天气预报助手",
        description="查询天气和画图",
        system_message=system,
        function_list=tools,
    )

    return bot


def test(
    query="海淀区天气", file: Optional[str] = os.path.join(ROOT_RESOURCE, "poem.pdf")
):
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []

    if not file:
        messages.append({"role": "user", "content": query})
    else:
        messages.append({"role": "user", "content": [{"text": query}, {"file": file}]})

    for response in bot.run(messages):
        print("bot response:", response)


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        # Query example: 海淀区天气
        query = input("user question: ")
        # File example: resource/poem.pdf
        file = input("file url (press enter if no file): ").strip()
        if not query:
            print("user question cannot be empty！")
            continue
        if not file:
            messages.append({"role": "user", "content": query})
        else:
            messages.append(
                {"role": "user", "content": [{"text": query}, {"file": file}]}
            )

        response = []
        for response in bot.run(messages):
            print("bot response:", response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        "prompt.suggestions": [
            "查询北京的天气",
            "画一张北京的图片",
            "画一张北京的图片，然后配上一首诗",
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == "__main__":
    # test()
    # app_tui()
    app_gui()
