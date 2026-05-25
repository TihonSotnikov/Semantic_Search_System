import asyncio
import logging
import argparse
import os
import json

import ranx
import torch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine, AsyncSession
from sqlalchemy import select, update, insert, delete
from sentence_transformers import SentenceTransformer

import app.ml.ml_engine as ml
import app.database.database as db
from app.logger.logger import configure_logging


logger = logging.getLogger('sss_evaluate')
ROOT = os.path.dirname(__file__)

queries = {
    "q_1": "Как работать из дома",
    "q_2": "Где заказать справку 2-НДФЛ",
    "q_3": "Оформление командировки и возврат денег",
    "q_4": "Как получить ДМС",
    "q_5": "Правила использования принтеров",
    "q_6": "Можно ли прийти на работу с собакой",
    "q_7": "Где оставить велосипед или самокат",
    "q_8": "Процесс увольнения и сдача техники",
    "q_9": "Что делать, если зависает или тормозит компьютер",
    "q_10": "Где взять почитать книги"
}

qrels_dict = {
    "q_1": {
        "1": 2,   # Основной документ про удаленную работу
        "28": 1   # Косвенно связанный документ про гибкий график
    },
    "q_2": {
        "5": 2,   # Заказ справок
        "82": 1   # Использование HR-портала общего назначения
    },
    "q_3": {
        "7": 2,   # Общие правила командировок
        "34": 2,  # Возмещение расходов
        "100": 2  # Порядок оформления расходов
    },
    "q_4": {
        "18": 2   # Полис ДМС
    },
    "q_5": {
        "12": 2,  # Использование принтера и МФУ
        "109": 2, # Ограничения по использованию принтеров
        "158": 2  # Работа с общими принтерами
    },
    "q_6": {
        "32": 2   # Dog-friendly офис
    },
    "q_7": {
        "25": 2,  # Велопарковка на паркинге
        "162": 2  # Правила парковки велосипедов
    },
    "q_8": {
        "46": 2,  # Процесс увольнения (Offboarding)
        "110": 2  # Обходной лист (Exit Checklist)
    },
    "q_9": {
        "20": 2,  # Оформление тикета в Helpdesk
        "42": 1,  # Плановое обновление техники
        "123": 1  # Профилактическая чистка ноутбука
    },
    "q_10": {
        "22": 2,  # Корпоративная библиотека
        "145": 2  # Использование офисной библиотеки
    }
}

qrels = ranx.Qrels(qrels_dict)


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
    logger.info('Creating embeddings for %d documents', len(documents))
    prompts = [f'# {doc['title']}\n{doc['text']}' for doc in documents]
    embeddings = ml.compute_embeddings(prompts, model)
    
    knowledge_list = [
        db.Knowledge(title=doc['title'], text=doc['text'], vector=emb)
        for doc, emb in zip(documents, embeddings)
    ]
    logger.debug('Generated %d knowledge records', len(knowledge_list))
    return knowledge_list

async def database_init(
        model: SentenceTransformer,
        session_maker: async_sessionmaker[AsyncSession]
        ):
    with open(os.path.join(ROOT, '../data/data_indexed.json'), encoding='utf8') as file:
        initial_data: list[dict] = json.load(file)

    knowledge_list = embeddings_from_docs(initial_data, model)
    async with session_maker() as session:
        stmt = delete(db.Knowledge)
        await session.execute(stmt)
        session.add_all(knowledge_list)
        await session.commit()

async def search(
        text: str,
        k: int,
        model: SentenceTransformer,
        device,
        session_maker: async_sessionmaker[AsyncSession]
        ):
    query_embedding = ml.encode_query(model, text).to(device)
    top_k_heap = []
    async with session_maker() as session:
        result_stream = await session.stream_scalars(select(db.Knowledge))
        async for partition in result_stream.partitions(100):
            batch_ids = []
            batch_vectors = []
            for row in partition:
                batch_ids.append(row.id)
                batch_vectors.append(row.vector)
            if not batch_vectors:
                continue
            embeddings_tensor = torch.stack(batch_vectors).to(device)
            scores = ml.compute_batch_scores(query_embedding, embeddings_tensor)
            top_k_heap = ml.select_top_k(top_k_heap, scores, batch_ids, k)
    final_results = {str(i): score for score, i in top_k_heap}
    return final_results

async def evaluate(model_name):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        engine = create_async_engine('sqlite+aiosqlite:///eval_data.db')
        session_maker = async_sessionmaker(engine, expire_on_commit=False)
        logger.info("DB engine created")
        model = ml.load_model(model_name).to(device)
        logger.info("Model '%s' loaded successfully on '%s'", model_name, device)
    
        async with engine.begin() as conn:
            await conn.run_sync(db.Base.metadata.create_all)
        await database_init(model, session_maker)
        
        search_results: dict[str, dict] = {}
        for i, (key, query) in enumerate(queries.items(), 1):
            logger.info("Processing queries: %d/%d", i, len(queries))
            search_results[key] = await search(query, 10, model, device, session_maker)

        run = ranx.Run(search_results)

        metrics = ["recall@3", "recall@5", "ndcg@3", "ndcg@5", "mrr@5"]
        results = ranx.evaluate(qrels, run, metrics)
        logger.info("Evaluation results:\n%s", json.dumps(results, ensure_ascii=False, indent=2))

    except Warning as w:
        logger.warning(w)
    except Exception as e:
        logger.error("Failed to run evaluation:", exc_info=True, stack_info=True)


if __name__ == '__main__':
    configure_logging("sss_evaluate", "evaluation.log")

    parser = argparse.ArgumentParser('SSS Evaluation')
    parser.add_argument('--model', '-m', type=str, default='google/embeddinggemma-300m')
    args = parser.parse_args()

    asyncio.run(evaluate(args.model))

