from openai import AsyncOpenAI
import os
from typing import TypeVar, Type
from pydantic import BaseModel

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

T = TypeVar("T", bound=BaseModel)

openai = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)


async def call_model(
    messages: list[ChatCompletionMessageParam],
    model: str = "mistral:latest",
    temperature: float = 0.0,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: str | None = None,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    res = await openai.chat.completions.create(**kwargs)
    return res


async def call_model_structured(
    messages: list[ChatCompletionMessageParam],
    response_format: Type[T],
    model: str = "mistral:latest",
    temperature: float = 0.0,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: str | None = None,
) -> T:
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
    }

    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    res = await openai.chat.completions.parse(**kwargs)

    parsed = res.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse model response into expected schema")

    return parsed
