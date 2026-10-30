from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from app.db import close_db, init_db
from document import router as document_router
from chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


router = APIRouter(prefix="/api/v1", lifespan=lifespan)

router.include_router(document_router)
router.include_router(chat_router)
