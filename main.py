import logging
import os
import json
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select, update, insert, delete

import src.database.database as db
import src.ml.ml_engine as ml
from src.frontend import frontend


ROOT = os.path.dirname(__file__)
MODELS = {
    'gte': 'Alibaba-NLP/gte-multilingual-base',
    'default': 'cointegrated/rubert-tiny2',
    'gemma': 'google/embeddinggemma-300m'
    }
BATCH_SIZE = 100

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
app.mount('/static', StaticFiles(directory='src/frontend/static'), name='static')
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
        with open(os.path.join(ROOT, 'data/data.json'), encoding='utf8') as file:
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
async def add_document(title: str | None = None, text: str | None = None):
    if not (text and title):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'No text or title provided')
    async with session_maker() as session:
        vector = ml.compute_embeddings([text], model)[0]
        value = db.Knowledge(
            text = text,
            title = title,
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
async def search(request: Request, text: str | None = None, k: int = 3):
    if not text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'No text provided')
    
    query_embedding = ml.encode_query(model, text).to(device)
    
    top_k_heap = []

    async with session_maker() as session:
        result_stream = await session.stream_scalars(select(db.Knowledge))
        
        async for partition in result_stream.partitions(BATCH_SIZE):
            batch_texts = []
            batch_vectors = []
            
            for row in partition:
                batch_texts.append(row.text)
                batch_vectors.append(row.vector)
            
            if not batch_vectors:
                continue
            
            embeddings_tensor = torch.stack(batch_vectors).to(device)
            scores = ml.compute_batch_scores(query_embedding, embeddings_tensor)
            
            top_k_heap = ml.select_top_k(top_k_heap, scores, batch_texts, k)

    final_results = sorted(
        [(text, score) for score, text in top_k_heap], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return final_results
