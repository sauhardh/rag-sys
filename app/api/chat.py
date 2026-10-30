from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionMessageFunctionToolCall
import json

from app.core.booking import BOOKING_TOOL_DEFINITION, save_booking
from app.core.embeddings import generate_query_embedding
from app.core.query import condense_query
from app.core.redis_mem import add_chat_turn, get_chat_history
from .document import collection
from app.core.llm import call_model, openai

router = APIRouter()


SYSTEM_PROMPT = """You are a helpful assistant for document Q&A and interview scheduling.
1. When answering questions from documents, only use the provided context. If unsure, state that the documents do not have the answer.
2. You can schedule interviews using the 'book_interview' function. To book, you MUST collect:
   - Full Name
   - Email
   - Date
   - Time
If any of these 4 details are missing, politely ask the user for them. Only call the function once ALL 4 details are gathered.
"""


BOOKING_KEYWORDS = {
    "interview",
    "book",
    "schedule",
    "slot",
    "appointment",
    "date",
    "time",
}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[str] = []
    booking_created: bool = False


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest):
    session_id = payload.session_id.strip()
    user_msg = payload.message.strip()

    if not session_id or not user_msg:
        raise HTTPException(
            status_code=400, detail="session_id and message are required"
        )

    # 1. Find the history of chat by particular person
    history_msg: list[ChatCompletionMessageParam] = await get_chat_history(
        session_id, 10
    )

    # 2. Check the intent
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages.extend(history_msg)
    messages.append({"role": "user", "content": user_msg})

    first_call = await call_model(
        messages=messages,
        tools=[BOOKING_TOOL_DEFINITION],
        tool_choice="auto",
    )

    first_choice = first_call.choices[0].message

    # CASE A: Model invoked booking tool
    if first_choice.tool_calls:
        tool_call = first_choice.tool_calls[0]

        if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
            if tool_call.function.name == "book_interview":
                args = json.loads(tool_call.function.arguments)
                booking = await save_booking(session_id, args)

                reply = f"Your interview has been booked for {booking.name} on {booking.interview_date} at {booking.interview_time}. Confirmation sent to {booking.email}."

                await add_chat_turn(session_id, user_msg, reply)
                return ChatResponse(
                    session_id=session_id, response=reply, booking_created=True
                )

    # Case B: User is in the interview booking flow (model asking for missing slots)
    is_booking_intent = any(kw in user_msg.lower() for kw in BOOKING_KEYWORDS)
    if is_booking_intent and first_choice.content:
        reply = first_choice.content
        await add_chat_turn(session_id, user_msg, reply)
        return ChatResponse(
            session_id=session_id, response=reply, booking_created=False
        )

    # Case C: Model replied directly
    standalone_query: str = await condense_query(history_msg, user_msg)

    vector_query = generate_query_embedding(standalone_query)
    (sources, context) = collection.query_formatted(vector_query)

    rag_messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nRelevant Context from Documents:\n{context}",
        }
    ]
    rag_messages.extend(history_msg)
    rag_messages.append({"role": "user", "content": user_msg})
    rag_call = await call_model(messages=rag_messages, temperature=0.2)
    final_reply = rag_call.choices[0].message.content or "No response generated."

    await add_chat_turn(session_id, user_msg, final_reply)
    return ChatResponse(
        session_id=session_id,
        response=final_reply,
        sources=list(sources),
        booking_created=False,
    )
