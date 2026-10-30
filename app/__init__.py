from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.db import close_db, init_db
from app.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(lifespan=lifespan)
app.include_router(router=router)
