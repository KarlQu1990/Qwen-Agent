import re
import cn2an
import copy
import json5
from typing import Iterator, List

from qwen_agent.agents.assistant import Assistant
from qwen_agent.llm.schema import ASSISTANT, CONTENT, Message
from qwen_agent.gui import WebUI
from qwen_agent import Agent


PROMPT_TEMPLATE_ZH = """
你是一个写作助手，任务是依据参考资料，完成写作任务。
#参考资料：
{ref_doc}

写作标题是：{user_request}
大纲是：
{outline}

写作要求：
- 扩写第{index1}章第{index2}节第{index3}小点对应的内容：{capture}。
- 不要生成第{index1}章和第{index2}节的标题，仅生成第{index3}小点的标题。
- 每个小点负责撰写不同的内容，所以你不需要为了全面而涵盖之后的内容。
- 只依据给定的参考资料来写，不要引入其余知识。
- 严格按照 Markdown 格式书写，当前小点的标题为三级标题。
"""


class ExpandWriting(Agent):

    def _run(
        self,
        messages: List[Message],
        knowledge: str = "",
        outline: str = "",
        index1: str = "1",
        index2: str = "1",
        index3: str = "1",
        capture: str = "",
        capture_later: str = "",
        **kwargs,
    ) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)

        user_request = messages[-1][CONTENT]
        if isinstance(messages[-1][CONTENT], list):
            user_request = messages[-1][CONTENT][0].text

        prompt = PROMPT_TEMPLATE_ZH.format(
            ref_doc=knowledge,
            user_request=user_request,
            index1=index1,
            index2=index2,
            index3=index3,
            outline=outline,
            capture=capture,
        )
        if capture_later:
            prompt = prompt + "请在涉及 " + capture_later + " 时停止。"

        messages[-1][CONTENT] = prompt

        return self._call_llm(messages)


default_plan = """{"action1": "expand"}"""


def is_top_header(s):
    pattern = r"^第[一二三四五六七八九十]+章 "
    match = re.match(pattern, s)
    return match is not None


def is_sub_header(s):
    pattern = r"^第[一二三四五六七八九十]+节 "
    match = re.match(pattern, s)
    return match is not None


def is_leaf_header(s):
    pattern = r"^[一二三四五六七八九十]+、"
    match = re.match(pattern, s)
    return match is not None


class WriteFromScratch(Agent):

    def _run(
        self, messages: List[Message], knowledge: str = "", lang: str = "en", outline: str = ""
    ) -> Iterator[List[Message]]:
        response = [Message(ASSISTANT, f">\n> Use Default plans: \n{default_plan} \n\n")]
        yield response
        res_plans = json5.loads(default_plan)

        for plan_id in sorted(res_plans.keys()):
            plan = res_plans[plan_id]
            if plan == "expand":
                response.append(Message(ASSISTANT, ">\n> Writing Text: \n\n"))
                yield response

                outline_list_all = outline.split("\n")
                outline_list_with_idx = []
                index1 = 0
                index2 = 0
                index3 = 0
                for x in outline_list_all:
                    ignore = False
                    if is_top_header(x):
                        index1 += 1
                        index2 = 0
                        index3 = 0
                    elif is_sub_header(x):
                        index2 += 1
                        index3 = 0
                    elif is_leaf_header(x):
                        index3 += 1
                    else:
                        ignore = True

                    if not ignore:
                        outline_list_with_idx.append([index1, index2, index3, x])

                otl_num = len(outline_list_with_idx)
                for i, (index1, index2, index3, v) in enumerate(outline_list_with_idx):
                    response.append(Message(ASSISTANT, ">\n# \n\n"))
                    yield response

                    header_level = 1
                    if index3 > 0:
                        header_level = 3
                    elif index2 > 0:
                        header_level = 2

                    capture = v.strip()
                    capture_later = ""
                    if i < otl_num - 1:
                        capture_later = outline_list_with_idx[i + 1][-1].strip()

                    if header_level == 1:
                        res_exp = [[Message(ASSISTANT, f"# {v}\n\n")]]
                    elif header_level == 2:
                        res_exp = [[Message(ASSISTANT, f"## {v}\n\n")]]
                    else:
                        exp_agent = ExpandWriting(llm=self.llm)
                        res_exp = exp_agent.run(
                            messages=messages,
                            knowledge=knowledge,
                            outline=outline,
                            index1=cn2an.an2cn(str(index1), "low"),
                            index2=cn2an.an2cn(str(index2), "low"),
                            index3=cn2an.an2cn(str(index3), "low"),
                            capture=capture,
                            capture_later=capture_later,
                            lang=lang,
                        )

                    chunk = None
                    for chunk in res_exp:
                        yield response + chunk
                    if chunk:
                        response.extend(chunk)
            else:
                pass


class ArticleAgent(Assistant):
    """This is an agent for writing articles.

    It can write a thematic essay or continue writing an article based on reference materials
    """

    def _run(self, messages: List[Message], lang: str = "zh", outline: str = "", **kwargs) -> Iterator[List[Message]]:

        # Need to use Memory agent for data management
        *_, last = self.mem.run(messages=messages, **kwargs)
        _ref = last[-1][CONTENT]

        response = []
        if _ref:
            response.append(Message(ASSISTANT, f">\n> Search for relevant information: \n{_ref}\n"))
            yield response

        writing_agent = WriteFromScratch(llm=self.llm, outline=outline)

        for rsp in writing_agent.run(messages=messages, lang=lang, knowledge=_ref, outline=outline):
            if rsp:
                yield response + rsp


def app_gui():
    generate_cfg = {"max_tokens": 20000, "max_input_tokens": 20000}
    llm_cfg = {
        "host_base_url": "http://192.168.201.86:9221/v2.1/models",
        "model": "Qwen2-7B-Instruct",
        "model_type": "bisheng",
        "generate_cfg": generate_cfg,
    }

    with open("outline.txt", "r", encoding="utf-8") as f:
        outline = f.read()

    bot = ArticleAgent(
        llm=llm_cfg,
        name="文章撰写助手",
        description="可以帮助撰写文章。",
    )

    WebUI(bot).run(full_article=True, outline=outline)


if __name__ == "__main__":
    app_gui()
