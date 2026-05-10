import logging
import os
import json
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.engine import CursorResult
from sqlalchemy import select, update, insert, delete
from pydantic import BaseModel, Field

import database.database as db
import ml.ml_engine as ml
from frontend import frontend


ROOT = os.path.dirname(__file__)
MODELS = {
    'gte': 'Alibaba-NLP/gte-multilingual-base',
    'default': 'cointegrated/rubert-tiny2',
    'gemma': 'google/embeddinggemma-300m'
    }
BATCH_SIZE = 100

class DocumentSchema(BaseModel):
    title: str = Field(..., min_length=3, max_length=40)
    text: str = Field(..., min_length=20, max_length=2000)

class SearchResult(BaseModel):
    score: float = Field(...)
    title: str = Field(...)
    text: str = Field(...)


engine = create_async_engine('sqlite+aiosqlite:///data.db')
session_maker = async_sessionmaker(engine, expire_on_commit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ml.load_model(MODELS['gemma']).to(device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = logging.getLogger('uvicorn')
    
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)

    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='frontend/static'), name='static')
app.include_router(frontend.router)


# @app.get('/') # , response_class=HTMLResponse
# async def index(request: Request):
#     return {
#         'message': 'Success'
#     }
#     # context = {
#     #     "request": request,
#     #     "title": 'DataBase Panel',
#     # }
#     # return templates.TemplateResponse(name="index.html", context=context)

@app.post('/reset')
async def database_reset(request: Request):
    logger: logging.Logger = request.app.state.logger
    try:
        with open(os.path.join(ROOT, '../data/data.json'), encoding='utf8') as file:
            initial_data: list = json.load(file)
            with open(os.path.join(ROOT, 'DEBUG.json'), 'w', encoding='utf8') as f:
                f.writelines(json.dumps(initial_data))
            # logger.info(f'Initial data:\n{initial_data}')

        texts = [doc['text'] for doc in initial_data]
        embeddings = ml.compute_embeddings(texts, model)
        
        knowledge_list = [
            db.Knowledge(title=doc['title'], text=doc['text'], vector=emb)
            for doc, emb in zip(initial_data, embeddings)
        ]
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
async def add_document(schema: DocumentSchema):
    async with session_maker() as session:
        vector = ml.compute_embeddings([schema.text], model)[0]
        value = db.Knowledge(
            text = schema.text,
            title = schema.title,
            vector = vector
        )

        session.add(value)
        await session.commit()
        return status.HTTP_200_OK

@app.delete('/delete_document')
async def delete_document(id: int):
    async with session_maker() as session:
        stmt = delete(db.Knowledge).where(db.Knowledge.id == id)
        result: CursorResult = await session.execute(stmt) # type: ignore
        await session.commit()
        return status.HTTP_200_OK if (result.rowcount > 0) else status.HTTP_304_NOT_MODIFIED


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

@app.get('/search', response_model=list[SearchResult])
async def search(request: Request, text: str = Query(...), k: int = 3):
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'No text provided')
    
    logger: logging.Logger = request.app.state.logger
    query_embedding = ml.encode_query(model, text).to(device)
    
    top_k_heap = []

    async with session_maker() as session:
        result_stream = await session.stream_scalars(select(db.Knowledge))
        
        async for partition in result_stream.partitions(BATCH_SIZE):
            batch_data = []
            batch_vectors = []
            
            for row in partition:
                batch_data.append({'title': row.title, 'text': row.text})
                batch_vectors.append(row.vector)
            
            if not batch_vectors:
                continue
            
            embeddings_tensor = torch.stack(batch_vectors).to(device)
            scores = ml.compute_batch_scores(query_embedding, embeddings_tensor)
            
            top_k_heap = ml.select_top_k(top_k_heap, scores, batch_data, k)

    final_results = sorted(
        [SearchResult(score=score, title=data['title'], text=data['text']) for score, data in top_k_heap], 
        key=lambda x: x.score, 
        reverse=True
    )
    logger.info('Results response:\n' + '\n'.join([f'{res.score}: {res.title}' for res in final_results]))
    
    return final_results
