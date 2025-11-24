import openai
import logging
import random
import time
import json
import asyncio
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
    cast,
    Callable
)

from autogen_core.base import CancellationToken
from autogen_core.components.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    UserMessage
)
from autogen_core.components.tools import Tool, ToolSchema
from autogen_ext.models import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from .wrapper import ClientWrapper


try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False

@dataclass
class Choice:
    index: int
    option: Any
    p: float

class Bandit(ABC):
    @abstractmethod
    def choose(self, options: Sequence[Any], context: Optional[Dict[str, Any]] = None, featurizer: Optional[Callable[[Any],  Dict[str, Any]]] = None) -> Choice:
        ...
    
    @abstractmethod
    def learn(self, options: Sequence[Any], choice : Choice, reward: float, context: Optional[Dict[str, Any]] = None, featurizer: Optional[Callable[[Any],  Dict[str, Any]]] = None):
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

class UniformExploration(Bandit):
    def choose(self, options: Sequence[Any], context: Optional[Dict[str, Any]] = None, featurizer: Optional[Callable[[Any],  Dict[str, Any]]] = None) -> Choice:
        index = np.random.randint(len(options))
        return Choice(index = index, option = options[index], p = 1 / len(options))
    
    def learn(self, options: Sequence[str], choice : Choice, reward: float, context: Optional[Dict[str, Any]] = None, featurizer: Optional[Callable[[Any],  Dict[str, Any]]] = None):
        ...

    @property
    def description(self):
        return "UniformExploration"
    
class GracefulRetryClient(ChatCompletionClient):
    """
        This class is a gateway to multiple clients, it will try to use the clients in a round robin fashion if one of them fails
        Every time create() is called, a new client is selected from the list of clients
    """
    def __init__(
        self,
        clients: List[ClientWrapper],
        support_json = True, # if all clients do
        logger: Optional[Type[logging.Logger]] = None,
        max_retries: int = 8,
        max_tokens: int = 50000,
        router = None,
        timeout: Optional[float] = None,
    ):
        self._clients = clients
        self._router = router or UniformExploration()
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._feat = lambda client, messages: {'desc': client.description, 'len': client.count_tokens(messages=messages)}
        self.support_json = support_json
        self.blocklist = set()

    def convert_client_type(self, new_client_type: Type[ChatCompletionClient]) -> None:
        for client in self._clients:
            client.convert_client_type(new_client_type)

    @staticmethod
    def from_files(files: Sequence[os.PathLike], logger, eval_model: str = 'gpt-4o'):
        client_jsons = []
        for client_config in files:
            with open(client_config) as f:
                config_openai = json.load(open(client_config))
                model_name = config_openai["CHAT_COMPLETION_KWARGS_JSON"]["model"]
                
                # breakpoint()

                # Only include models that match the specified eval_model type
                should_include = False
                if eval_model == '*':
                    should_include = True  # Include all models
                elif eval_model == 'gpt-4o':
                    should_include = (("gpt-4o" in model_name or "gpt4o" in model_name) and "o1" not in model_name and "o3" not in model_name and "o4" not in model_name)
                elif eval_model == 'o4-mini':
                    should_include = "o4" in model_name
                elif eval_model == 'o3-mini':
                    should_include = ("o3" in model_name and "mini" in model_name)
                elif eval_model == 'o3':
                    should_include = ("o3" in model_name and "mini" not in model_name)
                elif eval_model == 'gpt-5':
                    should_include = (("gpt-5" in model_name or "gpt5" in model_name) and "o1" not in model_name and "o3" not in model_name and "o4" not in model_name)
                
                if should_include:
                    client_jsons.append(config_openai)
        clients = [ClientWrapper.from_config(config=client_config) for client_config in client_jsons]
        if not clients:
            raise ValueError(f"Error! None of the models in the input judge config files match the type --eval_model={eval_model} in {files}")
        return GracefulRetryClient(clients=clients, logger=logger)

    @staticmethod
    def from_path(path: os.PathLike, logger, eval_model: str = 'gpt-4o'):
        endpoint_config = Path(path).resolve()
        if not endpoint_config.exists():
            raise ValueError(f"Endpoint config file {endpoint_config} does not exist.")
        
        if endpoint_config.is_dir():
            endpoint_config = list(endpoint_config.iterdir())
        else:
            endpoint_config = [endpoint_config]

        logger.info(f"loaded {len(endpoint_config)} endpoint configuration files: {path}")
        client_group = GracefulRetryClient.from_files(files = endpoint_config, logger = logger, eval_model = eval_model)
        logger.info(f"Instantiated {len(client_group._clients)} clients for the {eval_model} endpoints") 
        return client_group         
    
    def supports_json(self) -> bool:
        return self.support_json
    
    def next_client(self) -> ClientWrapper:
        """
            Self-healing property: only select from clients that are not in the blocklist, blocklist grows whenever we encounter a client that is not available
        """
        valid_clients = [client for client in self._clients if client.endpoint not in self.blocklist]
        idx = random.choice(list(range(len(valid_clients))))
        return valid_clients[idx]

    async def close(self):
        for client in self._clients:
            if hasattr(client, "close"):
                await client.close()
            elif hasattr(client, "_client"):
                if hasattr(client._client, "close"):
                    await client._client.close()

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        _feat = lambda client: self._feat(client, messages)
        tries = self.max_retries
        last_error = None
            
        while tries > 0:
            choice = None
            # if tries < self.max_retries:
            client = self.next_client()
            # else:
            #     choice = self._router.choose(options = self._clients, featurizer = _feat)
            #     client = self._clients[choice.index]
            request_tokens = client.count_tokens(messages=messages)
            if (not ("o3" in client.description or "o4" in client.description or "o1" in client.description)
                and "reasoning_effort" in extra_create_args):
                del extra_create_args["reasoning_effort"]
            # if "o3" in client.description or "o4-mini" in client.description:
            #     extra_create_args["reasoning_effort"] = "high"
            # else:
            #     if "reasoning_effort" in extra_create_args:
            #         del extra_create_args["reasoning_effort"]
            self.logger.info(f"GracefulRetryClient.create(): {client.description}, request_tokens: {request_tokens}, extra_create_args={extra_create_args}")
            if request_tokens and request_tokens > self.max_tokens:
                tries = 0
                self.logger.error(f"PromptTooLargeError: Requesting {request_tokens} tokens exceeding {self.max_tokens} is forbidden -- will shut down endpoint -- so abandoning request")
                raise RuntimeError(f"PromptTooLargeError: Requesting {request_tokens} tokens exceeding {self.max_tokens} is forbidden -- will shut down endpoint -- so abandoning request")
             
            try:
                start_time = time.time()
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        client.create(
                            messages=messages,
                            tools=tools,
                            json_output=json_output,
                            extra_create_args=extra_create_args,
                            cancellation_token=cancellation_token,
                        ),
                        timeout=self.timeout
                    )
                else:
                    result = await client.create(
                        messages=messages,
                        tools=tools,
                        json_output=json_output,
                        extra_create_args=extra_create_args,
                        cancellation_token=cancellation_token,
                    )
                if tries == self.max_retries:
                    latency = time.time() - start_time
                    reward = max(0, 25 - latency) / 25
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = reward)
                return result
            except asyncio.TimeoutError:
                tries -= 1
                self.logger.error(f"GracefulRetryClient.create() TimeoutError: {client.description} timed out after {self.timeout}s, switching to new client")
                print(f"GracefulRetryClient.create() TimeoutError: {client.description} timed out after {self.timeout}s, switching to new client")
                time.sleep(1)
                continue
            except openai.BadRequestError as e:
                if "Invalid prompt" in str(e) or "content management policy" in str(e) or "Please try again with a different prompt" in str(e):
                    self.logger.error(f"GracefulRetryClient.create() Invalid prompt: {client.description} Switching to new client;\n{e}")
                    tries -= 1
                    await asyncio.sleep(1)
                    last_error = e ### this may or may not be avoidable, but we want to try all clients before giving up
                    continue
                else:
                    self.logger.error(f"GracefulRetryClient.create() {client.description} Raising openai.BadRequestError: {e}")
                    raise e
            except openai.InternalServerError as e:
                self.logger.error(f"GracefulRetryClient.create() Caught InternalServerError: {client.description}, Switching to new client bc OpenAI API internal server error: {e}")
                print(f"GracefulRetryClient.create() Caught InternalServerError: {client.description}, Switching to new client bc OpenAI API internal server error: {e}")
                tries -= 1
                sleep_time = 2
                await asyncio.sleep(sleep_time)
                continue
            except openai.RateLimitError as e:
                tries -= 1
                sleep_time = 2 ** (self.max_retries - tries)
                self.logger.error(f"GracefulRetryClient.create() RateLimitError: {client.description}, switching to new client bc OpenAI API request exceeded rate limit, sleeping {sleep_time}s: {e}")
                if tries == self.max_retries:
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                await asyncio.sleep(sleep_time)
                continue
            except openai.NotFoundError as e:
                self.logger.error(f"GracefulRetryClient.create() NotFoundError: {client.description} does not exist, switching to new client {e}")
                if tries == self.max_retries:
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                self.blocklist.add(client.endpoint)
                print(f"ERROR: GracefulRetryClient.create() PermissionDeniedError: {client.description}, BLOCKING {client.endpoint} and switching to new client {e}")
                await asyncio.sleep(1)
                continue
            except openai.PermissionDeniedError as e:
                self.logger.error(f"GracefulRetryClient.create() PermissionDeniedError: {client.description}, switching to new client {e}")
                if tries == self.max_retries:
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                self.blocklist.add(client.endpoint)
                print(f"ERROR: GracefulRetryClient.create() PermissionDeniedError: {client.description}, BLOCKING {client.endpoint} and switching to new client {e}")
                await asyncio.sleep(1)
                continue
            except openai.APIConnectionError as e:
                self.logger.error(f"GracefulRetryClient.create() APIConnectionError: {client.description}, Switching to new client bc OpenAI API connection error: {e}")
                if tries == self.max_retries:
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                self.blocklist.add(client.endpoint)
                print(f"ERROR: GracefulRetryClient.create() APIConnectionError: {client.description}, BLOCKING {client.endpoint} and switching to new client {e}")
                await asyncio.sleep(1)
                continue
            except openai.AuthenticationError as e:
                self.logger.error(f"GracefulRetryClient.create() AuthenticationError: {client.description}, Switching to new client bc OpenAI API authentication error: {e}")
                if tries == self.max_retries:
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                ### do not penalize tries
                continue
            except openai.APIStatusError as e:
                if tries == self.max_retries:
                    self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                if "Prompt is too large" in str(e):
                    self.logger.error(f"GracefulRetryClient.create() PromptTooLargeError: ({request_tokens} tokens) is too big even though we checked it was less than {self.max_tokens}?\n{e}")
                if "DeploymentNotFound" in str(e):
                    self.logger.error(f"GracefulRetryClient.create() DeploymentNotFound: {client.description} does not exist, BLOCKING {client.endpoint} and switching to new client {e}")
                    if tries == self.max_retries:
                        self._router.learn(options = self._clients, featurizer = _feat, choice = choice, reward = -2)
                    self.blocklist.add(client.endpoint)
                    await asyncio.sleep(1)
                    continue
                if "Request body too large" in str(e):
                    ### unlike prompt is too large error, TODO @Corby suspects this is temporary in some endpoints that were rebooted, so try another endpint
                    self.logger.error(f"GracefulRetryClient.create() Request body too large: {client.description} Switching to new client;\n{e}")
                    tries -= 1
                    await asyncio.sleep(1)
                    continue
                raise e
            except Exception as e:
                if "please try again" in str(e).lower():
                    tries -= 1
                    self.logger.error(f"GracefulRetryClient.create() Caught Generic Exception using {client.description}, Switching to new client bc OpenAI API internal server error: {e}")
                    sleep_time = 2 ** (self.max_retries - tries)
                    await asyncio.sleep(sleep_time)
                    continue
                self.logger.error(f"GracefulRetryClient.create() {client.description} Raising Exception: {e}")
                print(f"Error: GracefulRetryClient.create() {client.description} Raising Exception: {e}")
                raise e
        if last_error:
            raise last_error
        valid_clients = [client for client in self._clients if client.endpoint not in self.blocklist]
        raise Exception(f"GracefulRetryClient.create(): All clients are exhausted even after {self.max_retries} retries to {len(valid_clients)}/{len(self._clients)} clients. Blocklist: {len(self.blocklist)}")
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'LlmClient': self._router.description if len(self._clients) > 1 else self._clients[0].description
        }


class ResponsesGracefulRetryClient(GracefulRetryClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_idx = random.randint(0, len(self._clients) - 1)

    @staticmethod
    def from_files(files: Sequence[os.PathLike], logger, eval_model: str = 'gpt-4o'):
        client_jsons = []
        for client_config in files:
            with open(client_config) as f:
                config_openai = json.load(open(client_config))
                model_name = config_openai["CHAT_COMPLETION_KWARGS_JSON"]["model"]
                
                # breakpoint()

                # Only include models that match the specified eval_model type
                should_include = False
                if eval_model == '*':
                    should_include = True  # Include all models
                elif eval_model == 'gpt-4o':
                    should_include = (("gpt-4o" in model_name or "gpt4o" in model_name) and "o1" not in model_name and "o3" not in model_name and "o4" not in model_name)
                elif eval_model == 'o4-mini':
                    should_include = "o4" in model_name
                elif eval_model == 'o3-mini':
                    should_include = ("o3" in model_name and "mini" in model_name)
                elif eval_model == 'o3':
                    should_include = ("o3" in model_name and "mini" not in model_name)
                elif eval_model == 'gpt-5':
                    should_include = (("gpt-5" in model_name or "gpt5" in model_name) and "o1" not in model_name and "o3" not in model_name and "o4" not in model_name)
                else:
                    raise ValueError(f"Unknown model type: {eval_model}")
                
                if should_include:
                    client_jsons.append(config_openai)
        clients = [ClientWrapper.from_config(config=client_config) for client_config in client_jsons]
        if not clients:
            raise ValueError(f"Error! None of the models in the input judge config files match the type --eval_model={eval_model} in {files}")
        return ResponsesGracefulRetryClient(clients=clients, logger=logger)

    @staticmethod
    def from_path(path: os.PathLike, logger, eval_model: str = 'gpt-4o'):
        endpoint_config = Path(path).resolve()
        if not endpoint_config.exists():
            raise ValueError(f"Endpoint config file {endpoint_config} does not exist.")
        
        if endpoint_config.is_dir():
            endpoint_config = list(endpoint_config.iterdir())
        else:
            endpoint_config = [endpoint_config]

        logger.info(f"loaded {len(endpoint_config)} endpoint configuration files: {path}")
        client_group = ResponsesGracefulRetryClient.from_files(files = endpoint_config, logger = logger, eval_model = eval_model)
        logger.info(f"Instantiated {len(client_group._clients)} clients for the {eval_model} endpoints") 
        return client_group

    def next_client(self, no_increment=False) -> ClientWrapper:
        valid_clients = [client for client in self._clients if client.endpoint not in self.blocklist]
        client = valid_clients[self._client_idx]
        if not no_increment:
            self._client_idx = (self._client_idx + 1) % len(valid_clients)
        return client

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        tries = self.max_retries
        last_error = None
        client = self.next_client(no_increment=True)
            
        while tries > 0:
            request_tokens = client.count_tokens(messages=messages)
            if (not ("o3" in client.description or "o4" in client.description or "o1" in client.description)
                and "reasoning_effort" in extra_create_args):
                del extra_create_args["reasoning_effort"]
            # if "o3" in client.description or "o4-mini" in client.description:
            #     extra_create_args["reasoning_effort"] = "high"
            # else:
            #     if "reasoning_effort" in extra_create_args:
            #         del extra_create_args["reasoning_effort"]
            self.logger.info(f"{self.__class__.__name__}.create(): {client.description}, request_tokens: {request_tokens}, extra_create_args={extra_create_args}")
            if request_tokens and request_tokens > self.max_tokens:
                tries = 0
                self.logger.error(f"PromptTooLargeError: Requesting {request_tokens} tokens exceeding {self.max_tokens} is forbidden -- will shut down endpoint -- so abandoning request")
                raise RuntimeError(f"PromptTooLargeError: Requesting {request_tokens} tokens exceeding {self.max_tokens} is forbidden -- will shut down endpoint -- so abandoning request")

            try:
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        client.create(
                            messages=messages,
                            tools=tools,
                            json_output=json_output,
                            extra_create_args=extra_create_args,
                            cancellation_token=cancellation_token,
                        ),
                        timeout=self.timeout
                    )
                else:
                    result = await client.create(
                        messages=messages,
                        tools=tools,
                        json_output=json_output,
                        extra_create_args=extra_create_args,
                        cancellation_token=cancellation_token,
                    )
                return result
            except asyncio.TimeoutError:
                tries -= 1
                client = self.next_client()
                self.logger.error(f"{self.__class__.__name__}.create() TimeoutError: {client.description} timed out after {self.timeout}s, switching to new client")
                time.sleep(1)
                continue
            except openai.BadRequestError as e:
                if "Invalid prompt" in str(e) or "content management policy" in str(e) or "Please try again with a different prompt" in str(e):
                    self.logger.error(f"{self.__class__.__name__}.create() Invalid prompt: {client.description} Switching to new client;\n{e}")
                    tries -= 1
                    client = self.next_client()
                    await asyncio.sleep(1)
                    last_error = e ### this may or may not be avoidable, but we want to try all clients before giving up
                    continue
                else:
                    self.logger.error(f"{self.__class__.__name__}.create() {client.description} Raising openai.BadRequestError: {e}")
                    raise e
            except openai.InternalServerError as e:
                self.logger.error(f"{self.__class__.__name__}.create() Caught InternalServerError: {client.description}, Switching to new client bc OpenAI API internal server error: {e}")
                tries -= 1
                sleep_time = 2
                client = self.next_client()
                await asyncio.sleep(sleep_time)
                continue
            except openai.RateLimitError as e:
                tries -= 1
                sleep_time = 2 ** (self.max_retries - tries)
                client = self.next_client()
                self.logger.error(f"{self.__class__.__name__}.create() RateLimitError: {client.description}, switching to new client bc OpenAI API request exceeded rate limit, sleeping {sleep_time}s: {e}")
                await asyncio.sleep(sleep_time)
                continue
            except openai.NotFoundError as e:
                self.logger.error(f"{self.__class__.__name__}.create() NotFoundError: {client.description}, BLOCKING {client.endpoint} and switching to new client {e}")
                self.blocklist.add(client.endpoint)
                client = self.next_client()
                await asyncio.sleep(1)
                continue
            except openai.PermissionDeniedError as e:
                self.logger.error(f"{self.__class__.__name__}.create() PermissionDeniedError: {client.description}, BLOCKING {client.endpoint} and switching to new client {e}")
                self.blocklist.add(client.endpoint)
                client = self.next_client()
                await asyncio.sleep(1)
                continue
            except openai.APIConnectionError as e:
                self.logger.error(f"{self.__class__.__name__}.create() APIConnectionError: {client.description}, BLOCKING {client.endpoint} and switching to new client bc OpenAI API connection error: {e}")
                self.blocklist.add(client.endpoint)
                client = self.next_client()
                await asyncio.sleep(1)
                continue
            except openai.AuthenticationError as e:
                self.logger.error(f"{self.__class__.__name__}.create() AuthenticationError: {client.description}, Switching to new client bc OpenAI API authentication error: {e}")
                ### do not penalize tries
                client = self.next_client()
                continue
            except openai.APIStatusError as e:
                if "Prompt is too large" in str(e):
                    self.logger.error(f"{self.__class__.__name__}.create() PromptTooLargeError: ({request_tokens} tokens) is too big even though we checked it was less than {self.max_tokens}?\n{e}")
                if "DeploymentNotFound" in str(e):
                    self.logger.error(f"{self.__class__.__name__}.create() DeploymentNotFound: {client.description} does not exist, BLOCKING {client.endpoint} and switching to new client {e}")
                    self.blocklist.add(client.endpoint)
                    client = self.next_client()
                    await asyncio.sleep(1)
                    continue
                if "Request body too large" in str(e):
                    ### unlike prompt is too large error, TODO @Corby suspects this is temporary in some endpoints that were rebooted, so try another endpint
                    self.logger.error(f"{self.__class__.__name__}.create() Request body too large: {client.description} Switching to new client;\n{e}")
                    tries -= 1
                    client = self.next_client()
                    await asyncio.sleep(1)
                    continue
                raise e
            except Exception as e:
                if "please try again" in str(e).lower():
                    tries -= 1
                    client = self.next_client()
                    self.logger.error(f"{self.__class__.__name__}.create() Caught Generic Exception using {client.description}, Switching to new client bc OpenAI API internal server error: {e}")
                    sleep_time = 2 ** (self.max_retries - tries)
                    await asyncio.sleep(sleep_time)
                    continue
                self.logger.error(f"{self.__class__.__name__}.create() {client.description} Raising Exception: {e}")
                raise e
        if last_error:
            raise last_error
        valid_clients = [client for client in self._clients if client.endpoint not in self.blocklist]
        raise Exception(f"{self.__class__.__name__}.create(): All clients are exhausted even after {self.max_retries} retries to {len(valid_clients)}/{len(self._clients)} clients. Blocklist: {len(self.blocklist)}")


class AnthropicGracefulRetryClient(ChatCompletionClient):
    """
    A graceful retry client for Anthropic endpoints.
    Similar to GracefulRetryClient but handles Anthropic-specific errors.
    This class is a gateway to multiple Anthropic clients, using them in a round robin fashion.
    """
    def __init__(
        self,
        clients: List[ClientWrapper],
        support_json = True,
        logger: Optional[Type[logging.Logger]] = None,
        max_retries: int = 8,
        max_tokens: int = 50000,
        router: Optional[Bandit] = None,
        timeout: Optional[float] = None,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK is not installed. Please install it with: pip install anthropic")

        self._clients = clients
        self._router = router or UniformExploration()
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._feat = lambda client, messages: {'desc': client.description, 'len': client.count_tokens(messages=messages)}
        self.support_json = support_json
        self.blocklist = set()

    def convert_client_type(self, new_client_type: Type[ChatCompletionClient]) -> None:
        for client in self._clients:
            client.convert_client_type(new_client_type)

    @staticmethod
    def from_files(files: Sequence[os.PathLike], logger, eval_model: str = 'claude-sonnet-4'):
        client_jsons = []
        for client_config in files:
            with open(client_config) as f:
                config = json.load(f)
                # Get model name from nested CHAT_COMPLETION_KWARGS_JSON
                kwargs_json = config.get("CHAT_COMPLETION_KWARGS_JSON", {})
                model_name = kwargs_json.get("model", "")

                # Filter models based on eval_model
                should_include = False
                if eval_model == '*':
                    should_include = True
                elif eval_model == 'claude-sonnet-4':
                    should_include = "claude-sonnet-4" in model_name or "claude-4-sonnet" in model_name
                elif eval_model == 'claude-opus-4':
                    should_include = "claude-opus-4" in model_name or "claude-4-opus" in model_name
                elif eval_model.startswith('claude-'):
                    should_include = eval_model in model_name
                else:
                    raise ValueError(f"Unknown Anthropic model type: {eval_model}")

                if should_include:
                    client_jsons.append(config)

        clients = [ClientWrapper.from_config(config=client_config) for client_config in client_jsons]
        if not clients:
            raise ValueError(f"Error! None of the models in the input config files match the type --eval_model={eval_model} in {files}")
        return AnthropicGracefulRetryClient(clients=clients, logger=logger)

    @staticmethod
    def from_path(path: os.PathLike, logger, eval_model: str = 'claude-sonnet-4'):
        endpoint_config = Path(path).resolve()
        if not endpoint_config.exists():
            raise ValueError(f"Endpoint config file {endpoint_config} does not exist.")

        if endpoint_config.is_dir():
            endpoint_config = list(endpoint_config.iterdir())
        else:
            endpoint_config = [endpoint_config]

        logger.info(f"loaded {len(endpoint_config)} endpoint configuration files: {path}")
        client_group = AnthropicGracefulRetryClient.from_files(files=endpoint_config, logger=logger, eval_model=eval_model)
        logger.info(f"Instantiated {len(client_group._clients)} clients for the {eval_model} endpoints")
        return client_group

    def supports_json(self) -> bool:
        return self.support_json

    def next_client(self) -> ClientWrapper:
        """
        Self-healing property: only select from clients that are not in the blocklist
        """
        valid_clients = [client for client in self._clients if client.endpoint not in self.blocklist]
        idx = random.choice(list(range(len(valid_clients))))
        return valid_clients[idx]

    async def close(self):
        for client in self._clients:
            if hasattr(client, "close"):
                await client.close()
            elif hasattr(client, "_client"):
                if hasattr(client._client, "close"):
                    await client._client.close()

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        _feat = lambda client: self._feat(client, messages)
        tries = self.max_retries
        last_error = None

        while tries > 0:
            choice = None
            client = self.next_client()
            request_tokens = client.count_tokens(messages=messages)

            self.logger.info(f"AnthropicGracefulRetryClient.create(): {client.description}, request_tokens: {request_tokens}, extra_create_args={extra_create_args}")
            if request_tokens and request_tokens > self.max_tokens:
                tries = 0
                self.logger.error(f"PromptTooLargeError: Requesting {request_tokens} tokens exceeding {self.max_tokens} is forbidden")
                raise RuntimeError(f"PromptTooLargeError: Requesting {request_tokens} tokens exceeding {self.max_tokens} is forbidden")

            try:
                start_time = time.time()
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        client.create(
                            messages=messages,
                            tools=tools,
                            json_output=json_output,
                            extra_create_args=extra_create_args,
                            cancellation_token=cancellation_token,
                        ),
                        timeout=self.timeout
                    )
                else:
                    result = await client.create(
                        messages=messages,
                        tools=tools,
                        json_output=json_output,
                        extra_create_args=extra_create_args,
                        cancellation_token=cancellation_token,
                    )
                if tries == self.max_retries:
                    latency = time.time() - start_time
                    reward = max(0, 25 - latency) / 25
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=reward)
                return result
            except asyncio.TimeoutError:
                tries -= 1
                self.logger.error(f"AnthropicGracefulRetryClient.create() TimeoutError: {client.description} timed out after {self.timeout}s")
                print(f"AnthropicGracefulRetryClient.create() TimeoutError: {client.description} timed out after {self.timeout}s")
                time.sleep(1)
                continue
            except anthropic.BadRequestError as e:
                if "Invalid" in str(e) or "content" in str(e):
                    self.logger.error(f"AnthropicGracefulRetryClient.create() Invalid prompt: {client.description};\n{e}")
                    tries -= 1
                    await asyncio.sleep(1)
                    last_error = e
                    continue
                else:
                    self.logger.error(f"AnthropicGracefulRetryClient.create() {client.description} Raising BadRequestError: {e}")
                    raise e
            except anthropic.InternalServerError as e:
                self.logger.error(f"AnthropicGracefulRetryClient.create() InternalServerError: {client.description}, Switching to new client: {e}")
                print(f"AnthropicGracefulRetryClient.create() InternalServerError: {client.description}, Switching to new client: {e}")
                tries -= 1
                await asyncio.sleep(2)
                continue
            except anthropic.RateLimitError as e:
                tries -= 1
                sleep_time = 2 ** (self.max_retries - tries)
                self.logger.error(f"AnthropicGracefulRetryClient.create() RateLimitError: {client.description}, sleeping {sleep_time}s: {e}")
                if tries == self.max_retries:
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=-2)
                await asyncio.sleep(sleep_time)
                continue
            except anthropic.NotFoundError as e:
                self.logger.error(f"AnthropicGracefulRetryClient.create() NotFoundError: {client.description}, BLOCKING {client.endpoint}: {e}")
                if tries == self.max_retries:
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=-2)
                self.blocklist.add(client.endpoint)
                print(f"ERROR: AnthropicGracefulRetryClient.create() NotFoundError: {client.description}, BLOCKING {client.endpoint}: {e}")
                await asyncio.sleep(1)
                continue
            except anthropic.PermissionDeniedError as e:
                self.logger.error(f"AnthropicGracefulRetryClient.create() PermissionDeniedError: {client.description}, BLOCKING {client.endpoint}: {e}")
                if tries == self.max_retries:
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=-2)
                self.blocklist.add(client.endpoint)
                print(f"ERROR: AnthropicGracefulRetryClient.create() PermissionDeniedError: {client.description}, BLOCKING {client.endpoint}: {e}")
                await asyncio.sleep(1)
                continue
            except anthropic.APIConnectionError as e:
                self.logger.error(f"AnthropicGracefulRetryClient.create() APIConnectionError: {client.description}, BLOCKING {client.endpoint}: {e}")
                if tries == self.max_retries:
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=-2)
                self.blocklist.add(client.endpoint)
                print(f"ERROR: AnthropicGracefulRetryClient.create() APIConnectionError: {client.description}, BLOCKING {client.endpoint}: {e}")
                await asyncio.sleep(1)
                continue
            except anthropic.AuthenticationError as e:
                self.logger.error(f"AnthropicGracefulRetryClient.create() AuthenticationError: {client.description}: {e}")
                if tries == self.max_retries:
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=-2)
                continue
            except anthropic.APIStatusError as e:
                if tries == self.max_retries:
                    self._router.learn(options=self._clients, featurizer=_feat, choice=choice, reward=-2)
                if "too large" in str(e).lower():
                    self.logger.error(f"AnthropicGracefulRetryClient.create() PromptTooLargeError: ({request_tokens} tokens)\n{e}")
                if "not found" in str(e).lower():
                    self.logger.error(f"AnthropicGracefulRetryClient.create() NotFound: {client.description}, BLOCKING {client.endpoint}: {e}")
                    self.blocklist.add(client.endpoint)
                    await asyncio.sleep(1)
                    continue
                if "body too large" in str(e).lower():
                    self.logger.error(f"AnthropicGracefulRetryClient.create() Request body too large: {client.description};\n{e}")
                    tries -= 1
                    await asyncio.sleep(1)
                    continue
                raise e
            except Exception as e:
                if "please try again" in str(e).lower():
                    tries -= 1
                    self.logger.error(f"AnthropicGracefulRetryClient.create() Generic Exception: {client.description}: {e}")
                    sleep_time = 2 ** (self.max_retries - tries)
                    await asyncio.sleep(sleep_time)
                    continue
                self.logger.error(f"AnthropicGracefulRetryClient.create() {client.description} Raising Exception: {e}")
                print(f"Error: AnthropicGracefulRetryClient.create() {client.description} Raising Exception: {e}")
                raise e

        if last_error:
            raise last_error
        valid_clients = [client for client in self._clients if client.endpoint not in self.blocklist]
        raise Exception(f"AnthropicGracefulRetryClient.create(): All clients exhausted after {self.max_retries} retries to {len(valid_clients)}/{len(self._clients)} clients. Blocklist: {len(self.blocklist)}")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'LlmClient': self._router.description if len(self._clients) > 1 else self._clients[0].description
        }
