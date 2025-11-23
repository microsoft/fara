"""
Adapter to wrap the official Anthropic Python SDK to work with autogen 0.4.0's ChatCompletionClient interface.
This allows us to use Anthropic models without upgrading to autogen 0.7.5+.
"""
import anthropic
import json
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Sequence, Union
from autogen_core.base import CancellationToken
from autogen_core.components.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.components.tools import Tool, ToolSchema
from autogen_core.components import FunctionCall


class AnthropicAdapter(ChatCompletionClient):
    """
    Adapter class that wraps the official Anthropic Python SDK to work with
    autogen 0.4.0's ChatCompletionClient interface.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        model_capabilities: Optional[ModelCapabilities] = None,
        max_retries: int = 5,
        **kwargs
    ):
        """
        Initialize the Anthropic adapter.

        Args:
            model: The Anthropic model ID (e.g., "claude-sonnet-4-5-20250929")
            api_key: Your Anthropic API key
            model_capabilities: Model capabilities (function_calling, json_output, vision)
            max_retries: Maximum number of retries for failed requests
            **kwargs: Additional arguments to pass to the Anthropic client
        """
        self._model = model
        self._api_key = api_key
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self._sync_client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        self._model_capabilities = model_capabilities or ModelCapabilities(
            vision=True,
            function_calling=True,
            json_output=True,
        )
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._create_args = kwargs

    def _convert_messages(self, messages: Sequence[LLMMessage]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert autogen LLMMessage format to Anthropic's message format.
        Returns (system_prompt, messages_list)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Anthropic uses a separate system parameter
                system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif isinstance(msg, UserMessage):
                content = msg.content
                # Handle multi-modal content (text + images)
                if isinstance(content, list):
                    formatted_content = []
                    for item in content:
                        if isinstance(item, str):
                            formatted_content.append({"type": "text", "text": item})
                        elif isinstance(item, dict):
                            if item.get("type") == "image_url":
                                # Convert image URL format
                                formatted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": item["image_url"]["url"] if isinstance(item.get("image_url"), dict) else item["image_url"]
                                    }
                                })
                            else:
                                formatted_content.append(item)
                    anthropic_messages.append({"role": "user", "content": formatted_content})
                else:
                    anthropic_messages.append({"role": "user", "content": str(content)})
            elif isinstance(msg, AssistantMessage):
                content = msg.content
                message_dict = {"role": "assistant"}

                # Handle function calls
                if hasattr(msg, 'function_calls') and msg.function_calls:
                    tool_calls = []
                    for fc in msg.function_calls:
                        tool_calls.append({
                            "type": "tool_use",
                            "id": fc.id,
                            "name": fc.name,
                            "input": fc.arguments
                        })
                    message_dict["content"] = tool_calls
                else:
                    message_dict["content"] = str(content) if content else ""

                anthropic_messages.append(message_dict)

        return system_prompt, anthropic_messages

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """
        Create a completion using the Anthropic API.
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request parameters
        request_params: Dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": extra_create_args.get("max_tokens", 4096),
        }

        if system_prompt:
            request_params["system"] = system_prompt

        # Handle tools/function calling
        if tools:
            anthropic_tools = []
            for tool in tools:
                if isinstance(tool, dict):
                    # Tool is already a dictionary
                    anthropic_tools.append({
                        "name": tool.get("name", tool.get("function", {}).get("name")),
                        "description": tool.get("description", tool.get("function", {}).get("description")),
                        "input_schema": tool.get("parameters", tool.get("function", {}).get("parameters", {}))
                    })
                elif isinstance(tool, Tool):
                    anthropic_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters
                    })
                else:
                    # ToolSchema or similar object
                    anthropic_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters
                    })
            request_params["tools"] = anthropic_tools

        # Handle JSON output mode
        if json_output:
            # Anthropic doesn't have a direct JSON mode like OpenAI,
            # but we can add it to the system prompt
            if system_prompt:
                request_params["system"] = system_prompt + "\n\nPlease respond with valid JSON only."
            else:
                request_params["system"] = "Please respond with valid JSON only."

        # Add any extra parameters, handling tool_choice conversion from OpenAI to Anthropic format
        for key, value in extra_create_args.items():
            if key == "tool_choice":
                # Convert OpenAI-style tool_choice to Anthropic format
                if isinstance(value, str):
                    if value == "auto":
                        request_params["tool_choice"] = {"type": "auto"}
                    elif value in ["required", "any"]:
                        request_params["tool_choice"] = {"type": "any"}
                    elif value == "none":
                        # Omit tool_choice parameter (Anthropic default is auto)
                        pass
                    else:
                        # Specific tool name
                        request_params["tool_choice"] = {"type": "tool", "name": value}
                elif isinstance(value, dict):
                    # Already in correct format, or OpenAI object format
                    if "type" in value:
                        # Already Anthropic format
                        request_params["tool_choice"] = value
                    elif "function" in value and "name" in value.get("function", {}):
                        # OpenAI format: {"type": "function", "function": {"name": "..."}}
                        request_params["tool_choice"] = {"type": "tool", "name": value["function"]["name"]}
            else:
                request_params[key] = value

        # Make the API call
        response = await self._client.messages.create(**request_params)

        # Update usage statistics
        usage = RequestUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens
        )
        self._actual_usage = usage
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens
        )

        # Convert response to CreateResult
        content = ""
        function_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                function_calls.append(
                    FunctionCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input)
                    )
                )

        # Determine finish reason and content
        finish_reason = response.stop_reason or "stop"
        if finish_reason == "end_turn":
            finish_reason = "stop"
        elif finish_reason == "tool_use":
            finish_reason = "function_calls"
        elif finish_reason == "max_tokens":
            finish_reason = "length"

        # In autogen 0.4.0, function calls go in the content field when finish_reason is "function_calls"
        if function_calls:
            result_content = function_calls
        else:
            result_content = content

        return CreateResult(
            content=result_content,
            usage=usage,
            finish_reason=finish_reason,
            cached=False,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """
        Streaming is not implemented for this adapter.
        """
        raise NotImplementedError("Streaming is not supported by AnthropicAdapter")

    def actual_usage(self) -> RequestUsage:
        """Return the usage from the last API call."""
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        """Return the total usage across all API calls."""
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        """
        Estimate token count for messages.
        Note: This is a rough estimate. Anthropic's actual tokenization may differ.
        """
        # Simple estimation: ~4 characters per token
        total_chars = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, str):
                        total_chars += len(item)
                    elif isinstance(item, dict) and "text" in item:
                        total_chars += len(str(item["text"]))

        # Add tokens for tools
        for tool in tools:
            total_chars += len(str(tool.name)) + len(str(tool.description))

        return total_chars // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool | ToolSchema] = []) -> int:
        """
        Calculate remaining tokens based on model's context window.
        Claude models typically have 200K token context windows.
        """
        context_window = 200000  # Claude's context window
        used_tokens = self.count_tokens(messages, tools)
        return max(0, context_window - used_tokens)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return the model's capabilities."""
        return self._model_capabilities
