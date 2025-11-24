import io
import base64
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict
from PIL import Image


@dataclass
class LLMMessage:
    content: str | List[Dict[str, Any]]
    source: str = "user"


@dataclass
class SystemMessage(LLMMessage):
    def __init__(self, content: str, source: str = "system"):
        self.content = content
        self.source = source


@dataclass
class UserMessage(LLMMessage):
    def __init__(self, content: str | List[Dict[str, Any]], source: str = "user"):
        self.content = content
        self.source = source


@dataclass
class AssistantMessage(LLMMessage):
    def __init__(self, content: str, source: str = "assistant"):
        self.content = content
        self.source = source


@dataclass
class ImageObj:
    """Image wrapper for handling screenshots and images"""

    image: Image.Image

    @classmethod
    def from_pil(cls, image: Image.Image) -> "ImageObj":
        return cls(image=image)

    def to_base64(self) -> str:
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def resize(self, size: Tuple[int, int]) -> Image.Image:
        """Resize the image"""
        return self.image.resize(size)


@dataclass
class ModelResponse:
    """Response from model call"""

    content: str
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCall:
    """Represents a function call with arguments"""

    id: str
    name: str
    arguments: Dict[str, Any]


def message_to_openai_format(message: LLMMessage) -> Dict[str, Any]:
    """Convert our LLMMessage to OpenAI API format"""
    role = (
        "system"
        if isinstance(message, SystemMessage)
        else "assistant"
        if isinstance(message, AssistantMessage)
        else "user"
    )

    # Handle multimodal content (text + images)
    if isinstance(message.content, list):
        content_parts = []
        for item in message.content:
            if isinstance(item, ImageObj):
                # Convert image to base64 data URL
                base64_image = item.to_base64()
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )
            elif isinstance(item, str):
                content_parts.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                # Already in proper format
                content_parts.append(item)
        return {"role": role, "content": content_parts}
    else:
        # Simple text content
        return {"role": role, "content": message.content}


@dataclass
class WebSurferEvent:
    source: str
    message: str
    url: str
    action: str | None = None
    arguments: Dict[str, Any] | None = None
