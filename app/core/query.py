from openai.types.chat import ChatCompletionMessageParam
from .llm import call_model


async def condense_query(
    history: list[ChatCompletionMessageParam],
    latest_question: str,
) -> str:
    if not history:
        return latest_question

    history_str = "\n".join(
        f"{msg['role']}: {msg.get('content')}" for msg in history[-6:]
    )

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone, complete search query.
    If the question is already self-contained, return it unchanged.
    Do NOT answer the question. Only output the rephrased query.
    Chat History:
    {history_str}
    Follow-up Question: {latest_question}
    Standalone Query:
    """

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    response = await call_model(messages=messages, temperature=0.0)

    content = response.choices[0].message.content
    return latest_question if not content else content.strip()
