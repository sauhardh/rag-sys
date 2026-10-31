from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
import json
import asyncio

from app.core.booking import BOOKING_TOOL_DEFINITION, save_booking
from app.core.embeddings import generate_query_embedding
from app.core.query import IntentDecision, condense_query, detect_intent
from app.core.redis_mem import add_chat_turn, get_chat_history
from .document import collection
from app.core.llm import call_model

router = APIRouter()


SYSTEM_PROMPT = """You are a helpful assistant for document Q&A and interview scheduling.
When answering questions from documents, only use the provided context. If unsure, state that the documents do not have the answer.
If user has no histroy context and first time messaging -- Welcome the user.
"""

BOOKING_SYSTEM_PROMPT = """You are an interview scheduling assistant.
To book an interview, you MUST collect compulsorily:
1. Full Name
2. Email
3. Date (YYYY-MM-DD)
4. Time (HH:MM)
If any of these details are missing from the conversation, politely ask the user to provide them.
Only call the 'book_interview' function once ALL 4 details are provided.
"""


class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[str] = []
    booking_created: bool = False


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest):
    session_id = payload.session_id.strip()
    user_msg = payload.message.strip()
    document_id = payload.document_id.strip() if payload.document_id else None

    if not session_id or not user_msg:
        raise HTTPException(
            status_code=400, detail="session_id and message are required"
        )

    history_msg: list[ChatCompletionMessageParam] = await get_chat_history(
        session_id, 10
    )
    intent: IntentDecision = await detect_intent(user_msg)

    if intent.intent == "BOOKING":
        return await _handle_booking(session_id, user_msg, history_msg)

    return await _handle_qa(session_id, user_msg, history_msg, document_id)


async def _handle_booking(
    session_id: str,
    user_msg: str,
    history_msg: list[ChatCompletionMessageParam],
) -> ChatResponse:
    booking_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": BOOKING_SYSTEM_PROMPT},
        *history_msg,
        {"role": "user", "content": user_msg},
    ]
    response = await call_model(
        messages=booking_messages,
        tools=[BOOKING_TOOL_DEFINITION],
        tool_choice="auto",
    )
    choice = response.choices[0].message

    if choice.tool_calls:
        tool_call = choice.tool_calls[0]

        if isinstance(tool_call, ChatCompletionMessageToolCall):
            if tool_call.function.name == "book_interview":
                try:
                    args = json.loads(tool_call.function.arguments)
                    booking = await save_booking(session_id, args)
                    reply = (
                        f"Your interview has been booked for {booking.name} on "
                        f"{booking.interview_date} at {booking.interview_time}. "
                        f"Confirmation sent to {booking.email}."
                    )
                    await add_chat_turn(session_id, user_msg, reply)
                    return ChatResponse(
                        session_id=session_id, response=reply, booking_created=True
                    )
                except ValidationError as ve:
                    missing_or_invalid = [err["loc"][0] for err in ve.errors()]
                    fields_str = ", ".join(str(f) for f in missing_or_invalid)
                    reply = f"Could you please provide a valid {fields_str} to complete your interview booking?"
                    await add_chat_turn(session_id, user_msg, reply)
                    return ChatResponse(session_id=session_id, response=reply)
    reply = (
        choice.content
        or "Could you please share your name, email, date, and preferred time for the interview?"
    )
    await add_chat_turn(session_id, user_msg, reply)
    return ChatResponse(session_id=session_id, response=reply)


async def _handle_qa(
    session_id: str,
    user_msg: str,
    history_msg: list[ChatCompletionMessageParam],
    document_id: str | None = None,
) -> ChatResponse:
    sources: set[str] = set()
    context: str = ""

    # Only query Qdrant if a document_id is specified
    if document_id:
        standalone_query = await condense_query(history_msg, user_msg)
        vector_query = await asyncio.to_thread(generate_query_embedding, standalone_query)
        sources, context = collection.query_formatted(vector_query, document_id=document_id)

    if context:
        rag_system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Relevant Context from Document:\n"
            f"{context}"
        )
    else:
        # Normal conversation fallback when no document is linked or found
        rag_system_prompt = (
            "You are a helpful and polite AI assistant. Engage in natural conversation "
            "with the user. If they ask about a document or specific file, let them know "
            "they can upload a document or provide its document ID."
        )

    rag_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": rag_system_prompt},
        *history_msg,
        {"role": "user", "content": user_msg},
    ]

    rag_call = await call_model(messages=rag_messages, temperature=0.3)
    final_reply = (
        rag_call.choices[0].message.content
        or "I'm here to help! Feel free to ask a question or upload a document."
    )
    await add_chat_turn(session_id, user_msg, final_reply)
    return ChatResponse(
        session_id=session_id,
        response=final_reply,
        sources=list(sources) if context else [],
    )
