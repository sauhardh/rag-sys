import json
import os
from openai.types.chat import ChatCompletionMessageParam

import redis.asyncio as aioredis

REDIS_URL = os.environ["REDIS_URL"]
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

SESSION_TTL_SECONDS = 86400  # 24 hours retention


async def get_chat_history(
    session_id: str, limit: int = 10
) -> list[ChatCompletionMessageParam]:
    key = f"chat:{session_id}"
    raw_history = await redis_client.lrange(key, -limit, -1)
    return [json.loads(msg) for msg in raw_history]


async def add_chat_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    key = f"chat:{session_id}"
    pipe = redis_client.pipeline()

    pipe.rpush(key, json.dumps({"role": "user", "content": user_msg}))
    pipe.rpush(key, json.dumps({"role": "assistant", "content": assistant_msg}))

    pipe.expire(key, SESSION_TTL_SECONDS)
    await pipe.execute()
