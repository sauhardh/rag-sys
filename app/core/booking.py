from typing import Any

from app.db import AsyncSessionLocal
from app.db.models import InterviewBooking
from openai.types.chat import ChatCompletionToolParam


BOOKING_TOOL_DEFINITION: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "book_interview",
        "description": (
            "Book an interview appointment when the candidate has provided "
            "their full name, email address, interview date, and interview time. "
            "Do not call this tool if any required information is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Candidate's full name",
                },
                "email": {
                    "type": "string",
                    "description": "Valid email address",
                },
                "date": {
                    "type": "string",
                    "description": "Interview date in YYYY-MM-DD format",
                },
                "time": {
                    "type": "string",
                    "description": "Interview time in HH:MM 24-hour format",
                },
            },
            "required": ["name", "email", "date", "time"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


async def save_booking(session_id: str, args: dict[str, Any]):
    async with AsyncSessionLocal() as session:
        booking = InterviewBooking(
            session_id=session_id,
            name=args.get("name"),
            email=args.get("email"),
            interview_date=args.get("date"),
            interview_time=args.get("time"),
            status="confirmed",
        )

        session.add(booking)
        await session.commit()

        return booking
