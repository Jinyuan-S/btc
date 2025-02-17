from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy.ext.asyncio import create_async_engine
from app.config import settings

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(settings.MONGO_URL)
    app.mongodb = app.mongodb_client.btc
    app.mysql_engine = create_async_engine(settings.MYSQL_URL)

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    await app.mysql_engine.dispose()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/blocks")
async def get_blocks():
    blocks = await app.mongodb.blocks.find().to_list(length=100)
    return blocks

@app.get("/transactions")
async def get_transactions():
    transactions = await app.mongodb.transactions.find().to_list(length=100)