"""A data analysis example implemented by assistant"""

import os
from pprint import pprint
from typing import Optional

from qwen_agent.agents import ReActChat
from qwen_agent.gui import WebUI

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")


def init_agent_service():
    generate_cfg = {"max_tokens": 20000, "max_input_tokens": 20000}
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
        "generate_cfg": generate_cfg,
    }
    tools = ["retrieval"]
    bot = ReActChat(
        llm=llm_cfg,
        name="文档问答助手",
        description="基于文档回答问题",
        function_list=tools,
    )
    return bot


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        # Query example: pd.head the file first and then help me draw a line chart to show the changes in stock prices
        query = input("user question: ")
        # File example: resource/stock_prices.csv
        file = input("file url (press enter if no file): ").strip()
        if not query:
            print("user question cannot be empty！")
            continue
        if not file:
            messages.append({"role": "user", "content": query})
        else:
            messages.append({"role": "user", "content": [{"text": query}, {"file": file}]})

        response = []
        for response in bot.run(messages):
            print("bot response:", response)
        messages.extend(response)


def app_gui():
    bot = init_agent_service()
    WebUI(bot).run()


if __name__ == "__main__":
    # app_tui()
    app_gui()
