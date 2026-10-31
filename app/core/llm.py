from openai import AsyncOpenAI
from typing import TypeVar, Type
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)

openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)


async def call_model(
    messages: list[ChatCompletionMessageParam],
    model: str = settings.llm_model,
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

    res = await openai_client.chat.completions.create(**kwargs)
    return res


async def call_model_structured(
    messages: list[ChatCompletionMessageParam],
    response_format: Type[T],
    model: str = settings.llm_model,
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

    res = await openai_client.chat.completions.parse(**kwargs)

    parsed = res.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Failed to parse model response into expected schema")

    return parsed
