import logging
import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select, update, insert

import src.database.database as db
import src.ml.ml_engine as ml


engine = create_async_engine('sqlite+aiosqlite:///data.db')
session_maker = async_sessionmaker(engine, expire_on_commit=False)
model = ml.load_model()
templates = Jinja2Templates('frontend')


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = logging.getLogger('uvicorn')
    
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get('/') # , response_class=HTMLResponse
async def index(request: Request):
    return {
        'message': 'Success'
    }
    # context = {
    #     "request": request,
    #     "title": 'DataBase Panel',
    # }
    # return templates.TemplateResponse(name="index.html", context=context)

@app.post('/add_document')
async def add_document(text = None):
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'No text provided')
    async with session_maker() as session:
        vector = ml.compute_embeddings([text], model)[0]
        value = db.Knowledge(
            text = text,
            vector = vector
        )

        session.add(value)
        await session.commit()
        return status.HTTP_200_OK

@app.get('/dump')
async def dump_data(request: Request):
    async with session_maker() as session:
        stmt = select(db.Knowledge)
        result = await session.execute(stmt)
        result = result.scalars().all()
        logger: logging.Logger = request.app.state.logger
        if result:
            logger.info(f'Size of the 1st emb: {result[0].vector.shape}')
        return result
