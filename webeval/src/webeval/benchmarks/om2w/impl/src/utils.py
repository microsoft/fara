import base64
import io
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)
import os
import backoff

def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_predication(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebVoyager_eval":
        if "FAILURE" in response:
            return 0
        else:
            return 1
    elif mode == "WebJudge_Online_Mind2Web_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0  
    elif mode == "WebJudge_general_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0      
    else:
        raise ValueError(f"Unknown mode: {mode}")


class OpenaiEngine():
    def __init__(
        self,
        api_key=None,
        stop=[],
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri = "",
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.client = OpenAI(
                        api_key=api_key,
                    )

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=log_error
    )
    def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs):
        model = model if model else self.model
        response = self.client.chat.completions.create(
            model=model if model else self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        return [choice.message.content for choice in response.choices]
    