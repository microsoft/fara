import json
import requests
from typing import List, Dict, Any

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Literal
from urllib.parse import urlparse
import re
from copy import deepcopy

from autogen_core.application.logging.events import LLMCallEvent
from autogen_core.components import Image
from autogen_core.components.models import (
    ChatCompletionClient,
    ModelCapabilities,
)
from .systems.messages import (
    AgentEvent,
    AssistantContent,
    FunctionExecutionContent,
    OrchestrationEvent,
    SystemContent,
    UserContent,
    WebSurferEvent,
    TaskProposalEvent,
)
from autogen_ext.models import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient


ENVIRON_KEY_CHAT_COMPLETION_PROVIDER = "CHAT_COMPLETION_PROVIDER"
ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON = "CHAT_COMPLETION_KWARGS_JSON"

# The singleton _default_azure_ad_token_provider, which will be created if needed
_default_azure_ad_token_provider = None

def replace_url_with_netloc(text):
    """
        replace a string containing a URL with just the netloc, 
        example: "I visited https://www.tusd.org/educational-services/career-technical-education/cte-pathway-programs" -> "I visited www.tusd.org"
    """
    def replace_func(match):
        url = match.group(0)
        parsed = urlparse(url)
        return parsed.netloc

    return re.sub(r'https?://[^\s"]+', replace_func, text)

def attempt_parse_json(json_str: str) -> Dict[str, Any]:
    assert isinstance(json_str, str)
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    try:
        r = json.loads(json_str)
    except json.JSONDecodeError:
        r = eval(json_str)
    return r
    
# Create a model client based on information provided in environment variables.
def create_completion_client_from_env(env: Dict[str, str] | None = None, **kwargs: Any) -> ChatCompletionClient:
    global _default_azure_ad_token_provider

    """
    Create a model client based on information provided in environment variables.
        env (Optional):     When provied, read from this dictionary rather than os.environ
        kwargs**:           ChatClient arguments to override (e.g., model)

    NOTE: If 'azure_ad_token_provider' is included, and euquals the string 'DEFAULT' then replace it with
          azure.identity.get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    """
    # don't modify env, make a deep copy
    if env is None:
        env_copy = dict(os.environ)
    else:
        env_copy = deepcopy(env)
    
    
    # Load the kwargs, and override with provided kwargs
    _kwargs = env_copy.get(ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON, "{}")
    if isinstance(_kwargs, str):
        _kwargs = json.loads(_kwargs)
    _kwargs.update(kwargs)

    # If model capabilities were provided, deserialize them as well
    if "model_capabilities" in _kwargs:
        _kwargs["model_capabilities"] = ModelCapabilities(
            vision=_kwargs["model_capabilities"].get("vision"),
            function_calling=_kwargs["model_capabilities"].get("function_calling"),
            json_output=_kwargs["model_capabilities"].get("json_output"),
        )

    # Figure out what provider we are using. Default to OpenAI
    _provider = env_copy.get(ENVIRON_KEY_CHAT_COMPLETION_PROVIDER, "openai").lower().strip()
    assert _provider in ["openai", "azure", "trapi"]

    # Instantiate the correct client
    if _provider == "openai":
        _kwargs.pop('proxies', None)

        return OpenAIChatCompletionClient(**_kwargs)  # type: ignore
    elif _provider == "azure":
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential, ManagedIdentityCredential
        if _kwargs.get("azure_ad_token_provider", "").lower() == "default":
            if _default_azure_ad_token_provider is None:

                _default_azure_ad_token_provider = get_bearer_token_provider(
                    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
                )
            _kwargs["azure_ad_token_provider"] = _default_azure_ad_token_provider
        elif _kwargs.get("azure_ad_token_provider", "").lower() == "uami":
            if _default_azure_ad_token_provider is None:

                _default_azure_ad_token_provider = get_bearer_token_provider(
                    ManagedIdentityCredential(client_id=os.environ['AZURE_CLIENT_ID']), "https://cognitiveservices.azure.com/.default"
                )
            _kwargs["azure_ad_token_provider"] = _default_azure_ad_token_provider
        return AzureOpenAIChatCompletionClient(**_kwargs)  # type: ignore
    elif _provider == "trapi":
        assert _kwargs["api_key"] != "", "api_key (from TRAPI) must be set in environment variables"
        return AzureOpenAIChatCompletionClient(**_kwargs)  
    else:
        raise ValueError(f"Unknown OAI provider '{_provider}'")

def download_file(url: str, filepath: str) -> None:
    response = requests.get(url)
    response.raise_for_status()
    with open(filepath, 'wb') as file:
        file.write(response.content)

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def dict_2_str(d: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a string representation deterministically.
    """
    return '.'.join(f'{k}-{d[k]}' for k in sorted(d.keys()))

# Convert UserContent to a string
def message_content_to_str(
    message_content: UserContent | AssistantContent | SystemContent | FunctionExecutionContent,
) -> str:
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, dict):
        return json.dumps(message_content, indent=2)
    elif isinstance(message_content, List):
        converted: List[str] = list()
        for item in message_content:
            if isinstance(item, str):
                converted.append(item.rstrip())
            elif isinstance(item, Image):
                converted.append("<Image>")
            else:
                converted.append(str(item).rstrip())
        return "\n".join(converted)
    else:
        raise AssertionError("Unexpected response type.")

# MagenticOne log event handler
class LogHandler(logging.FileHandler):
    def __init__(self, filename: str = "log.jsonl") -> None:
        super().__init__(filename)
        self.logs_list: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).isoformat()
            if isinstance(record.msg, OrchestrationEvent):
                console_message = (
                    f"\n{'-'*75} \n" f"\033[91m[{ts}], {record.msg.source}:\033[0m\n" f"\n{record.msg.message}"
                )
                #print(console_message, flush=True)
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": record.msg.message,
                        "type": "OrchestrationEvent",
                    }
                )
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            elif isinstance(record.msg, AgentEvent):
                console_message = (
                    f"\n{'-'*75} \n" f"\033[91m[{ts}], {record.msg.source}:\033[0m\n" f"\n{record.msg.message}"
                )
                #print(console_message, flush=True)
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": record.msg.message,
                        "type": "AgentEvent",
                    }
                )
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            elif isinstance(record.msg, WebSurferEvent):
                console_message = f"\033[96m[{ts}], {record.msg.source}: {record.msg.message}\033[0m"
                #print(console_message, flush=True)
                payload: Dict[str, Any] = {
                    "timestamp": ts,
                    "type": "WebSurferEvent",
                }
                # if hasattr(record.msg, "source"):
                #     if record.msg.source != "screenshot_websurfer":
                #         print(console_message, flush=True)

                payload.update(asdict(record.msg))
                record.msg = json.dumps(payload)
                
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            elif isinstance(record.msg, LLMCallEvent):
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "prompt_tokens": record.msg.prompt_tokens,
                        "completion_tokens": record.msg.completion_tokens,
                        "type": "LLMCallEvent",
                    }
                )
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            else:
                try:
                    payload = asdict(record.msg)
                except Exception:
                    payload = {"message": str(record.msg)}
                payload["timestamp"] = ts
                payload["type"] = "OtherEvent"
                record.msg = json.dumps(payload)
                self.logs_list.append(json.loads(record.msg))
                super().emit(record)
            
        except Exception:
            self.handleError(record)    

