import json
from openai.types.chat import ChatCompletionMessageParam
import redis.asyncio as aioredis
import logging

from app.core.config import settings


redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
logger = logging.getLogger(__name__)


async def get_chat_history(
    session_id: str, limit: int = 10
) -> list[ChatCompletionMessageParam]:
    key = f"chat:{session_id}"

    try:
        raw_history = await redis_client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in raw_history]
    except Exception:
        logger.warning(
            "Redis unavailable — returning empty chat history for session %s",
            session_id,
        )
        return []


async def add_chat_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    key = f"chat:{session_id}"

    try:
        pipe = redis_client.pipeline()

        pipe.rpush(key, json.dumps({"role": "user", "content": user_msg}))
        pipe.rpush(key, json.dumps({"role": "assistant", "content": assistant_msg}))

        pipe.expire(key, settings.session_ttl_seconds)
        await pipe.execute()
    except Exception:
        logger.warning(
            "Redis unavailable — chat turn not persisted for session %s", session_id
        )
