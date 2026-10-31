from typing import Any
from pydantic import BaseModel, EmailStr
from datetime import date, time
from openai.types.chat import ChatCompletionToolParam
import logging

from app.db import AsyncSessionLocal
from app.db.models import InterviewBooking

logger = logging.getLogger(__name__)


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


class BookingInput(BaseModel):
    name: str
    email: EmailStr
    date: date
    time: time


async def save_booking(session_id: str, args: dict[str, Any]) -> InterviewBooking:
    booking_input = BookingInput.model_validate(args)

    async with AsyncSessionLocal() as session:
        booking = InterviewBooking(
            session_id=session_id,
            name=booking_input.name,
            email=booking_input.email,
            interview_date=booking_input.date,
            interview_time=booking_input.time,
            status="confirmed",
        )

        session.add(booking)
        await session.commit()
        await session.refresh(booking)

        logger.info("Booking created: %s for session %s", booking.id, session_id)
        return booking
