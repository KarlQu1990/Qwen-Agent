from qwen_agent.agents import ArticleAgent
from qwen_agent.gui import WebUI


def app_gui():
    generate_cfg = {"max_tokens": 20000}
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
        "generate_cfg": generate_cfg,
    }

    bot = ArticleAgent(
        llm=llm_cfg,
        name="文章撰写助手",
        description="可以帮助撰写文章。",
    )

    WebUI(bot).run(full_article=True)


if __name__ == "__main__":
    app_gui()
