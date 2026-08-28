"""Chat completion clients used by webeval (OpenAI / OrcaRouter / Azure OpenAI / Azure ML)."""

from .create_utils import (
    ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON,
    ENVIRON_KEY_CHAT_COMPLETION_PROVIDER,
    create_client_from_config,
    create_client_from_file,
    create_completion_client_from_env,
)
from .graceful_client import (
    GracefulRetryClient,
    ResponsesGracefulRetryClient,
)
from .messages import (
    AssistantMessage,
    CreateResult,
    FunctionCall,
    ImageObj,
    LLMMessage,
    RequestUsage,
    SystemMessage,
    Tool,
    ToolSchema,
    UserMessage,
    message_to_openai_format,
)
from .wrapper import (
    AzureMLClientWrapper,
    AzureOpenAIClientWrapper,
    AzureOpenAIResponsesWrapper,
    ChatCompletionClient,
    ClientWrapper,
    ModelCapabilities,
    OpenAIClientWrapper,
    OrcaRouterClientWrapper,
)

__all__ = [
    "AssistantMessage",
    "AzureMLClientWrapper",
    "AzureOpenAIClientWrapper",
    "AzureOpenAIResponsesWrapper",
    "ChatCompletionClient",
    "ClientWrapper",
    "CreateResult",
    "ENVIRON_KEY_CHAT_COMPLETION_KWARGS_JSON",
    "ENVIRON_KEY_CHAT_COMPLETION_PROVIDER",
    "FunctionCall",
    "GracefulRetryClient",
    "ImageObj",
    "LLMMessage",
    "ModelCapabilities",
    "OpenAIClientWrapper",
    "OrcaRouterClientWrapper",
    "RequestUsage",
    "ResponsesGracefulRetryClient",
    "SystemMessage",
    "Tool",
    "ToolSchema",
    "UserMessage",
    "create_client_from_config",
    "create_client_from_file",
    "create_completion_client_from_env",
    "message_to_openai_format",
]
