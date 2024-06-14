import requests
import json
from pprint import pformat
from typing import Dict, Optional, Iterator, List

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.text_base import BaseTextChatModel
from qwen_agent.log import logger

from .schema import ASSISTANT, Message


@register_llm("bisheng")
class BishengHostLLM(BaseTextChatModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or "gpt-3.5-turbo"
        cfg = cfg or {}

        host_base_url = cfg.get("host_base_url", "")

        url = host_base_url.rsplit("/", 2)[0]
        config_ep = f"{url}/v2/models/{self.model}/config"
        config = requests.get(url=config_ep, json={}, timeout=5).json()
        policy = config.get("model_transaction_policy", {})
        decoupled = policy.get("decoupled", False)

        if decoupled:
            self.host_base_url = f"{host_base_url}/{self.model}/generate_stream"
        else:
            self.host_base_url = f"{host_base_url}/{self.model}/infer"

        if cfg.get("headers", None):
            self.headers = cfg["headers"]
        else:
            self.headers = {"Content-Type": "application/json"}

        self.timeout = cfg.get("request_timeout", 20)

    def _chat_complete_create(self, **kwargs):
        params = {"stream": False}
        params.update(self.generate_cfg)
        params.update(kwargs)

        try:
            resp = requests.post(url=self.host_base_url, json=params, headers=self.headers, timeout=self.timeout)
            if resp.text.startswith("data:"):
                resp = json.loads(resp.text.replace("data:", ""))
            else:
                resp = resp.json()
        except requests.exceptions.Timeout as exc:
            raise ValueError(f"timeout in host llm infer, url=[{self.host_base_url}]") from exc
        except Exception as e:
            raise ValueError(f"exception in host llm infer: [{e}]") from e

        if not resp.get("choices", []):
            logger.info(resp)
            raise ValueError(f"empty choices in llm chat result {resp}")

        return resp

    def _chat_complete_create_stream(self, **kwargs):
        params = {"stream": True}
        params.update(self.generate_cfg)
        params.update(kwargs)

        print("######## params:", params)

        response = requests.post(url=self.host_base_url, json=params, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Error: {response.status_code} content: {response.text}")

        def iter_respone(response):
            for text in response.iter_content(chunk_size=1024, decode_unicode=True):
                if "\n" in text:
                    for text_ in text.split("\n"):
                        yield text_.strip()
                else:
                    yield text.strip()

        text_haf = ""
        for text in iter_respone(response):
            is_error = False
            if text:
                if text.startswith("event:error"):
                    is_error = True
                elif text.startswith("data:"):
                    text = text[len("data:") :].strip()
                    if text == "[DONE]":
                        break
                    try:
                        json.loads(text_haf + text)
                        yield (is_error, text_haf + text)
                        text_haf = ""
                    except Exception:
                        # 拆包了
                        if text_haf.startswith("{"):
                            text_haf = text
                            continue
                        logger.error(f"response_not_json response={text}")

                    if is_error:
                        break
                elif text.startswith("{"):
                    yield (is_error, text)
                else:
                    continue

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = [msg.model_dump() for msg in messages]
        logger.debug(f"*{pformat(messages, indent=2)}*")
        try:
            response = self._chat_complete_create_stream(model=self.model, messages=messages, **generate_cfg)
            if delta_stream:
                for is_error, chunk in response:
                    output = json.loads(chunk)
                    if is_error:
                        logger.error(chunk)
                        raise ValueError(chunk)

                    choices = output.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content", ""):
                        yield [Message(ASSISTANT, choices[0]["delta"]["content"])]
            else:
                full_response = ""
                for is_error, chunk in response:
                    output = json.loads(chunk)
                    if is_error:
                        logger.error(chunk)
                        raise ValueError(chunk)

                    choices = output.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content", ""):
                        full_response += choices[0]["delta"]["content"]
                        yield [Message(ASSISTANT, full_response)]

        except Exception as ex:
            logger.exception("error::")
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = [msg.model_dump() for msg in messages]
        logger.debug(f"*{pformat(messages, indent=2)}*")
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            choices = response.get("choices", [])
            return [Message(ASSISTANT, choices[0]["message"]["content"])]
        except Exception as ex:
            raise ModelServiceError(exception=ex)
