import uvicorn
from decouple import config
from fastapi import FastAPI

from api import api
from logger import LoggerSingleton

logger = LoggerSingleton("main").logger

app = FastAPI()
app.include_router(api)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config("host"),
        port=config("port", default=5000, cast=int),
        reload=config('DEBUG', default=False, cast=bool)
    )
