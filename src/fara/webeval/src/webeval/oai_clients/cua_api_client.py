import asyncio
import copy
import json
import logging
import warnings

from asyncio import Task
from dataclasses import dataclass, asdict
from PIL import Image
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union
)

from autogen_core.application.logging import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
from autogen_core.application.logging.events import LLMCallEvent
from autogen_core.base import CancellationToken
from autogen_core.components import Image as AGImage
from autogen_core.components.models import (
    ChatCompletionClient,
)
from autogen_core.components.models import (
    LLMMessage,
    RequestUsage,
    UserMessage,
)
from autogen_core.components.tools import Tool, ToolSchema
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_ext.models._openai._openai_client import (
    convert_tools,
    to_oai_type,
    _add_usage,
)
from openai.types.responses import (
    Response,
    response_create_params,
)


logger = logging.getLogger(EVENT_LOGGER_NAME)
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

create_kwargs = set(response_create_params.ResponseCreateParamsBase.__annotations__.keys()) | set(
    ("timeout", "stream")
)
# Only single choice allowed
disallowed_create_args = set(["stream", "messages", "function_call", "functions", "n"])
required_create_args: Set[str] = set(["model"])

@dataclass
class CUAResponse:
    id: str
    output: Dict[str, Any]
    usage: RequestUsage


@dataclass
class CUAResponseComputerCallOutput:
    call_id: str
    acknowledged_safety_checks: List[Any]
    output: Dict[str, Any]
    type: str = "computer_call_output"


class AzureOpenAICUAResponsesClient(AzureOpenAIChatCompletionClient):
    def _build_api_parameters(
            self,
            json_output: Optional[bool] = None,
            extra_create_args: Mapping[str, Any] = {},
        ) -> Dict[str, Any]:
        api_params: Dict[str, Any] = {}
        if json_output:
            api_params["text"] = {"type": "json_object"}
        api_params["truncation"] = "auto"
        api_params.update(extra_create_args)
        api_params["model"] = self._client._azure_deployment
        return api_params

    def count_tokens(
            self,
            messages: Sequence[Union[CUAResponse, CUAResponseComputerCallOutput, Dict[str, Any]]],
            tools: Sequence[Tool | ToolSchema] = [],
        ) -> int:
        return (len(messages) * 100)+ (len(tools) * 10) + 100  # Dummy implementation for illustration

    async def create(
            self,
            messages: Sequence[Union[CUAResponse, CUAResponseComputerCallOutput, Dict[str, Any]]],
            tools: Sequence[Tool | ToolSchema] = [],
            json_output: Optional[bool] = None,
            extra_create_args: Mapping[str, Any] = {},
            cancellation_token: Optional[CancellationToken] = None,
        ) -> CUAResponse:
            # Make sure all extra_create_args are valid
            extra_create_args_keys = set(extra_create_args.keys())
            if not create_kwargs.issuperset(extra_create_args_keys):
                raise ValueError(f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}")

            # Copy the create args and overwrite anything in extra_create_args
            create_args = self._create_args.copy()
            create_args.update(self._build_api_parameters(json_output, extra_create_args))

            # # TODO: allow custom handling.
            # # For now we raise an error if images are present and vision is not supported
            # if self.capabilities["vision"] is False:
            #     for message in messages:
            #         if isinstance(message, UserMessage):
            #             if isinstance(message.content, list) and any(isinstance(x, (Image, AGImage)) for x in message.content):
            #                 raise ValueError("Model does not support vision and image was provided")

            if json_output is not None:
                if self.capabilities["json_output"] is False and json_output is True:
                    raise ValueError("Model does not support JSON output")

                if json_output is True:
                    create_args["response_format"] = {"type": "json_object"}
                else:
                    create_args["response_format"] = {"type": "text"}

            if self.capabilities["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output")

            oai_messages = []
            for m in messages:
                tmp = response_to_oai(m)
                if isinstance(tmp, list):
                    oai_messages += tmp
                else:
                    oai_messages.append(tmp)

            if self.capabilities["function_calling"] is False and len(tools) > 0:
                raise ValueError("Model does not support function calling")

            future: Task[Response]
            if len(tools) > 0:
                converted_tools = []
                for tool in tools:
                    try:
                        converted_tools += convert_tools([tool])
                    except KeyError:
                        converted_tools.append(tool)  # assume already converted

                future = asyncio.ensure_future(
                    self._client.responses.create(
                        tools=converted_tools,
                        input=oai_messages,
                        stream=False,
                        **create_args,
                    )
                )
            else:
                future = asyncio.ensure_future(
                    self._client.responses.create(
                        input=oai_messages,
                        stream=False,
                        **create_args,
                    )
                )

            if cancellation_token is not None:
                cancellation_token.link_future(future)
            result: Response = await future

            if result.usage is not None:
                logger.info(
                    LLMCallEvent(
                        prompt_tokens=result.usage.input_tokens,
                        completion_tokens=result.usage.output_tokens,
                    )
                )

            usage = RequestUsage(
                # TODO backup token counting
                prompt_tokens=result.usage.input_tokens if result.usage is not None else 0,
                completion_tokens=(result.usage.output_tokens if result.usage is not None else 0),
            )

            if self._resolved_model is not None:
                if self._resolved_model != result.model:
                    warnings.warn(
                        f"Resolved model mismatch: {self._resolved_model} != {result.model}. Model mapping may be incorrect.",
                        stacklevel=2,
                    )

            result_json = json.loads(result.to_json())

            response = CUAResponse(
                id=result_json.get("id", "unknown"),
                output=result_json["output"],
                usage=usage,
            )

            _add_usage(self._actual_usage, usage)
            _add_usage(self._total_usage, usage)

            return response


def convert_to_oai_format(img: Union[Image.Image, AGImage]) -> str:
    if isinstance(img, AGImage):
        return img.to_openai_format()["image_url"]["url"]
    elif isinstance(img, Image.Image):
        tmp = AGImage.from_pil(img)
        return tmp.to_openai_format()["image_url"]["url"]


def response_to_oai(response: Union[CUAResponse, CUAResponseComputerCallOutput, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(response, CUAResponse):
        return response.output
    elif isinstance(response, CUAResponseComputerCallOutput):
        tmp = asdict(response)
        for k, v in tmp["output"].items():
            if isinstance(v, (Image.Image, AGImage)):
                tmp["output"][k] = convert_to_oai_format(v)
        return [tmp]
    elif isinstance(response, dict):
        tmp = copy.deepcopy(response)
        if isinstance(tmp["content"], dict):
            for k, v in tmp["content"].items():
                if isinstance(v, (Image.Image, AGImage)):
                    tmp["content"][k] = convert_to_oai_format(v)
            return tmp
        elif isinstance(tmp["content"], list):
            for i, item in enumerate(tmp["content"]):
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, (Image.Image, AGImage)):
                            item[k] = convert_to_oai_format(v)
                    tmp["content"][i] = item
            return [tmp]
        else:
            return [tmp]
    else:
        raise TypeError(f"Unsupported response type: {type(response)}")


if __name__ == "__main__":
    img = Image.open("/home/spwhitehead/src/agento/waa/src/win-arena-container/client/local_results/0/failed/data_1_0_1757007111721/task_solving/screenshot_1.png")
    # msgs = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Tag all photos in the "Summer Trip" folder with a custom tag "2023Vacation".'}, {'type': 'input_image', 'image_url': img}]}]
    msgs = [CUAResponse(id='resp_68ba4bdf6bc8819ebc2fe6712dfd162b0a42e0250e18906c', output=[{'id': 'rs_68ba4be0c0d4819e930e2794198617750a42e0250e18906c', 'summary': [], 'type': 'reasoning'}, {'id': 'cu_68ba4be19204819e883e0bf62806a3370a42e0250e18906c', 'action': {'keys': ['CTRL', 'A'], 'type': 'keypress'}, 'call_id': 'call_Br7Y65Lko5vohktkrnFLjYdy', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}], usage=RequestUsage(prompt_tokens=1631, completion_tokens=107)), CUAResponseComputerCallOutput(call_id='call_Br7Y65Lko5vohktkrnFLjYdy', acknowledged_safety_checks=[], output={'type': 'input_image', 'image_url': img, 'type': 'computer_call_output'})]
    oai_msgs = []
    for m in msgs:
        tmp = response_to_oai(m)
        if isinstance(tmp, list):
            oai_msgs += tmp
        else:
            oai_msgs.append(tmp)
    # oai_msgs = [response_to_oai(m) for m in msgs]

    print(oai_msgs)
