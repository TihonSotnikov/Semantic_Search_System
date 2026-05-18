import logging
import os
import json
import argparse
from contextlib import asynccontextmanager
from traceback import extract_tb
from typing import Any

import torch
import uvicorn
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, status, Query, UploadFile, File
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine
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


engine: AsyncEngine
model: SentenceTransformer
session_maker: async_sessionmaker

logger = logging.getLogger('uvicorn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
database_url = 'sqlite+aiosqlite:///data.db'
model_name = MODELS['gemma']


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, session_maker, model
    need_init = True
    if (os.path.exists(os.path.join(ROOT, 'data.db')) and database_url == 'sqlite+aiosqlite:///data.db'):
        need_init = False

    engine = create_async_engine(database_url)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    model = ml.load_model(model_name).to(device)
    
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)
    
    if need_init:
        await database_reset()

    yield
    await engine.dispose()

def embeddings_from_docs(documents: list[dict], model):
    """
    Объединяет заголовки с текстами из списка словарей,
    извлекает из них векторные представления и
    собирает список моделей, готовый к загрузке в базу данных.

    Parameters
    ----------
    documents : list[dict]
        Список словарей в формате
        `[{"title": "...", "text": "..."},]`
    model : SentenceTransformer
        Инициализированная модель векторизации текста

    Returns
    -------
    list[Knowledge]
        Список записей для базы данных
    """
    prompts = [f'# {doc['title']}\n{doc['text']}' for doc in documents]
    embeddings = ml.compute_embeddings(prompts, model)
    
    knowledge_list = [
        db.Knowledge(title=doc['title'], text=doc['text'], vector=emb)
        for doc, emb in zip(documents, embeddings)
    ]
    return knowledge_list


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='frontend/static'), name='static')
app.include_router(frontend.router)


@app.post('/reset')
async def database_reset():
    """
    Сброс базы знаний к состоянию по-умолчанию.
    Очищает базу и импортирует стандартные документы из `/data/data.json`.

    Returns
    -------
    Response
        Пустой ответ со статусом:
        - 200: Успешно

    Raises
    ------
    HTTPException
        - 500: Не удалось загрузить базу из `data.json`
    """
    try:
        with open(os.path.join(ROOT, '../data/data.json'), encoding='utf8') as file:
            initial_data: list = json.load(file)
    except Exception as e:
        logger.error(f'Error loading initial data: {e}')
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, 'Failed to load initial data')

    knowledge_list = embeddings_from_docs(initial_data, model)
    async with session_maker() as session:
        stmt = delete(db.Knowledge)
        await session.execute(stmt)
        session.add_all(knowledge_list)
        await session.commit()

    return Response(status_code=status.HTTP_200_OK)

@app.post('/clear')
async def database_clear(request: Request):
    logger: logging.Logger = request.app.state.logger

    async with session_maker() as session:
        stmt = delete(db.Knowledge)
        await session.execute(stmt)
        await session.commit()

    return Response(status_code=status.HTTP_200_OK)

@app.post('/import_data')
async def import_data(request: Request, files: list[UploadFile] = File(...)):
    logger: logging.Logger = request.app.state.logger
    files_failed = []

    async with session_maker() as session:
        for file in files:
            try:
                if file.content_type != 'application/json':
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, 'Invalid file type')
                document_list = json.loads(await file.read())
                
                knowledge_list = embeddings_from_docs(document_list, model)
                session.add_all(knowledge_list)
            except Exception as e:
                stack = extract_tb(e.__traceback__)
                tb_filename, tb_line_number, tb_func_name, tb_text = stack[-1]
                logger.error(f'Error processing file {file.filename} at {tb_filename}:{tb_line_number} in {tb_func_name}')
                files_failed.append(file.filename)
        await session.commit()
    
    if files_failed:
        response = JSONResponse(
            status_code=status.HTTP_207_MULTI_STATUS,
            content={'files_failed': files_failed}
        )
    else:
        response = Response(status_code=status.HTTP_200_OK)
        
    return response

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
    """
    Эндпоинт для одиночного удаления записи из базы знаний по id.

    Parameters
    ----------
    id : int
        Уникальный id записи.

    Returns
    -------
    _type_
        _description_
    """
    async with session_maker() as session:
        stmt = delete(db.Knowledge).where(db.Knowledge.id == id)
        result: CursorResult = await session.execute(stmt) # type: ignore
        await session.commit()
    if (result.rowcount > 0):
        return JSONResponse({"status": "OK"}, status_code=status.HTTP_200_OK)
    else:
        return JSONResponse({"status": "Запись не найдена"}, status_code=status.HTTP_204_NO_CONTENT)


@app.get('/dump')
async def dump_data(request: Request):
    """
    Эндпоинт для получения всей базы знаний
    в формате JSON.

    Returns
    -------
    JSONResponse
        Список словарей в JSON формате.
        `[ { "id": int, "title": str, "text": str } ]`
    """
    async with session_maker() as session:
        stmt = select(db.Knowledge)
        result = await session.execute(stmt)
        result = result.scalars().all()
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

def main():
    global model_name, database_url
    parser = argparse.ArgumentParser("Semantic Search System")
    parser.add_argument(
        '--database',
        type=str,
        default='sqlite+aiosqlite:///data.db',
        help="URL базы данных для sqlalchemy.",
        required=False
        )
    parser.add_argument(
        '--model',
        type=str,
        default='google/embeddinggemma-300m',
        help="Идентификатор модели векторизации текста с HF.",
        required=False
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help="Идентификатор модели векторизации текста с HF.",
        required=False
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help="Идентификатор модели векторизации текста с HF.",
        required=False
    )
    args = parser.parse_args()

    database_url = args.database
    model_name = args.model
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=True
    )
