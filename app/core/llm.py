from openai import AsyncOpenAI
import os

openai = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/api"),
)
