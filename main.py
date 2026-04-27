import logging
import datetime
import torch
import os
import json

from fastapi import FastAPI, HTTPException, status
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select, update, insert, delete

import src.database.database as db
import src.ml.ml_engine as ml


engine = create_async_engine('sqlite+aiosqlite:///data.db')
session_maker = async_sessionmaker(engine, expire_on_commit=False)
model = ml.load_model('Alibaba-NLP/gte-multilingual-base')
templates = Jinja2Templates('frontend')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(__file__)


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

@app.post('/reset')
async def database_reset(request: Request):
    logger: logging.Logger = request.app.state.logger
    try:
        with open(os.path.join(ROOT, 'data/data.json'), encoding='utf8') as file:
            initial_data: list = json.load(file)
            logger.info(f'Initial data:\n{initial_data}')
        knowledge_list = [
            db.Knowledge(vector=ml.compute_embeddings(doc['text'], model), **doc)
            for doc in initial_data]
        async with session_maker() as session:
            stmt = delete(db.Knowledge)
            await session.execute(stmt)
            session.add_all(knowledge_list)
            await session.commit()
    except HTTPException as e:
        return e
    except Exception as e:
        logger.error(e)
        return {'message': 'Internal error'}
    return {'message': 'Success'}

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

@app.get('/search')
async def search(request: Request, text = None):
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'No text provided')
    async with session_maker() as session:
        stmt = select(db.Knowledge)
        data = await session.scalars(stmt)
        data = data.all()

        docs = []
        embs = []
        for row in data:
            docs.append(row.text)
            embs.append(row.vector)
        embs = torch.stack(embs).to(device)
        results = ml.search_similar_texts(text, docs, embs, model)
        return results
