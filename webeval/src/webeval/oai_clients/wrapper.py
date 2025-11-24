import json
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Tuple,
    Mapping,
    Optional,
    Sequence,
    Callable,
    Set,
    Type,
    Union,
    cast,
)

from autogen_core.base import CancellationToken
from autogen_core.components.models import (
    ChatCompletionClient,
    RequestUsage,
    ModelCapabilities,
    CreateResult,
    LLMMessage,
)
from autogen_core.components.tools import Tool, ToolSchema
from autogen_ext.models import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from ..utils import create_completion_client_from_env, ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON, ENVIRON_KEY_CHAT_COMPLETION_PROVIDER

try:
    import anthropic
    from agento.oai_clients.anthropic_adapter import AnthropicAdapter
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    
class ClientWrapper(ChatCompletionClient):
    """
        Multiprocessing compatible wrapper of ChatCompletionClient with extra metadata required for debugging / logging of agento infra

    """
    def __init__(
        self,
        impl: ChatCompletionClient,
        metadata = None):
        self._impl = impl
        self._metadata = metadata or {}

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult: 
        return await self._impl.create(messages, tools, json_output, extra_create_args, cancellation_token)

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        return self._impl.create_stream(messages, tools, json_output, extra_create_args, cancellation_token)  

    def actual_usage(self) -> RequestUsage:
        return self._impl.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._impl.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._impl.count_tokens(messages, tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._impl.remaining_tokens(messages, tools)
    
    def convert_client_type(self, new_client_type: Type[ChatCompletionClient]) -> ChatCompletionClient:
        self._impl = cast_client(self._impl, new_client_type)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._impl.capabilities
    
    @property
    def metadata(self):
        return self._metadata

    @property
    def endpoint(self) -> str:
        #e.g. https://reasoning-eastus.openai.azure.com/gpt-4o-reasoning-51
        if 'azure_endpoint' in self.metadata and 'azure_deployment' in self.metadata:
            return f"{self.metadata['azure_endpoint']}{self.metadata['azure_deployment']}"
        elif 'model' in self.metadata:
            # For Anthropic or other non-Azure providers
            return self.metadata['model']
        else:
            return "unknown_endpoint"

    @property
    def description(self) -> str:
        if 'azure_endpoint' in self.metadata and 'azure_deployment' in self.metadata:
            return f"{self.metadata['azure_endpoint']}{self.metadata['azure_deployment']} - {self._impl._create_args}"
        elif 'model' in self.metadata:
            # For Anthropic or other non-Azure providers
            return f"{self.metadata['model']} - {self._impl._create_args}"
        else:
            return f"unknown - {self._impl._create_args}"  # TODO: fix abstraction leakage
    
    @staticmethod
    def from_file(path: str) -> 'ClientWrapper':
        with open(path, "r") as f:
            config = json.load(f)
        return ClientWrapper.from_config(config)
    
    @staticmethod
    def from_config(config: Dict) -> 'ClientWrapper':
        provider = config.get(ENVIRON_KEY_CHAT_COMPLETION_PROVIDER, "openai").lower().strip()
        kwargs_json = config[ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON]

        if provider == "anthropic":
            metadata = {
                'model': kwargs_json.get('model', 'unknown'),
                'provider': 'anthropic'
            }
        else:
            # Azure or OpenAI
            metadata = {
                'azure_endpoint': kwargs_json.get('azure_endpoint', ''),
                'azure_deployment': kwargs_json.get('azure_deployment', ''),
            }

        return ClientWrapper(
            impl = create_completion_client_from_env(config),
            metadata = metadata
        )  


def cast_client(client, tgt_cls):
    # create a fresh child instance with the same __dict__ as parent
    casted = tgt_cls.__new__(tgt_cls)   # bypass __init__
    casted.__dict__.update(client.__dict__) # copy state
    return casted
