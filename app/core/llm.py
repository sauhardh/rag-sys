from openai import AsyncOpenAI
import os

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

openai = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)


async def call_model(
    messages: list[ChatCompletionMessageParam],
    model: str = "llama3.2:1b",
    temperature: float = 0.0,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: str | None = None,
):
    if not messages:
        raise ValueError("messages parameter must not be empty while calling model")

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
