from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionMessageFunctionToolCall
import json

from app.core.booking import BOOKING_TOOL_DEFINITION, save_booking
from app.core.embeddings import generate_query_embedding
from app.core.query import IntentDecision, condense_query, detect_intent
from app.core.redis_mem import add_chat_turn, get_chat_history
from .document import collection
from app.core.llm import call_model

router = APIRouter()


SYSTEM_PROMPT = """You are a helpful assistant for document Q&A and interview scheduling.
When answering questions from documents, only use the provided context. If unsure, state that the documents do not have the answer.
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

    # Find the intent
    intent: IntentDecision = await detect_intent(user_msg)

    ## Interview Booking

    if intent.intent == "BOOKING":
        booking_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": BOOKING_SYSTEM_PROMPT},
        ]
        booking_messages.extend(history_msg)
        booking_messages.append({"role": "user", "content": user_msg})

        response = await call_model(
            messages=booking_messages,
            tools=[BOOKING_TOOL_DEFINITION],
            tool_choice="auto",
        )
        choice = response.choices[0].message

        # CASE A: Model called booking tool
        if choice.tool_calls:
            tool_call = choice.tool_calls[0]

            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                if tool_call.function.name == "book_interview":
                    args = json.loads(tool_call.function.arguments)
                    missing = [
                        f for f in ["name", "email", "date", "time"] if not args.get(f)
                    ]

                    if missing:
                        reply = f"Could you please also provide your {', '.join(missing)} to complete the booking?"

                        await add_chat_turn(session_id, user_msg, reply)
                        return ChatResponse(
                            session_id=session_id, response=reply, booking_created=False
                        )

                    # Persist booking to PostgreSQL
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

        # CASE B: Model called for more information about booking
        reply = (
            choice.content
            or "Could you please share your name, email, date, and preferred time for the interview?"
        )
        await add_chat_turn(session_id, user_msg, reply)
        return ChatResponse(
            session_id=session_id, response=reply, booking_created=False
        )

    ## QA
    standalone_query: str = await condense_query(history_msg, user_msg)
    vector_query = generate_query_embedding(standalone_query)
    sources, context = collection.query_formatted(vector_query)

    rag_system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Relevant Context from Documents:\n{context if context else 'No document context available.'}"
    )
    rag_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": rag_system_prompt},
    ]
    rag_messages.extend(history_msg)
    rag_messages.append({"role": "user", "content": user_msg})

    rag_call = await call_model(messages=rag_messages, temperature=0.2)
    final_reply = (
        rag_call.choices[0].message.content
        or "I could not find an answer in the provided documents."
    )

    await add_chat_turn(session_id, user_msg, final_reply)
    return ChatResponse(
        session_id=session_id,
        response=final_reply,
        sources=list(sources) if context else [],
        booking_created=False,
    )
