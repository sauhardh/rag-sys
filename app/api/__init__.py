from fastapi import APIRouter
from document import router as document_router
from chat import router as chat_router

router = APIRouter(prefix="/api/v1")

router.include_router(document_router)
router.include_router(chat_router)
