import torch
from sentence_transformers import SentenceTransformer, util
import heapq
from typing import List, Tuple


def load_model(model_name: str = "cointegrated/rubert-tiny2") -> SentenceTransformer:
    """
    Загружает и возвращает NLP-модель для генерации эмбеддингов.

    Parameters
    ----------
    model_name : str
        Идентификатор модели на HuggingFace.

    Returns
    -------
    SentenceTransformer
        Инициализированная модель.
    """

    return SentenceTransformer(model_name, trust_remote_code=True)


def compute_embeddings(texts: List[str], model: SentenceTransformer) -> torch.Tensor:
    """
    Вычисляет векторные представления для списка текстов.

    Parameters
    ----------
    texts : List[str]
        Массив строк для векторизации.
    model : SentenceTransformer
        Загруженная модель векторизации.

    Returns
    -------
    torch.Tensor
        Тензор эмбеддингов, оптимизированный для PyTorch.
    """

    return model.encode(texts, convert_to_tensor=True)


def encode_query(model: SentenceTransformer, query: str) -> torch.Tensor:
    """Кодирование запроса в вектор."""
    return model.encode(query, convert_to_tensor=True)


def compute_batch_scores(
    query_embedding: torch.Tensor, 
    batch_embeddings: torch.Tensor
) -> torch.Tensor:
    """Вычисление косинусного сходства для батча."""
    return util.cos_sim(query_embedding, batch_embeddings)[0]


def select_top_k(
    existing_top_k: List[Tuple[float, str]], 
    scores: torch.Tensor, 
    texts: List[str], 
    k: int
) -> List[Tuple[float, str]]:
    """
    Обновление списка top-k с использованием min-heap.
    Храним (score, text), чтобы heapq сравнивал по score.
    """
    for score, text in zip(scores.tolist(), texts):
        if len(existing_top_k) < k:
            heapq.heappush(existing_top_k, (score, text))
        else:
            if score > existing_top_k[0][0]:
                heapq.heapreplace(existing_top_k, (score, text))
    return existing_top_k


def search_similar_texts(
    query: str,
    corpus_texts: List[str],
    corpus_embeddings: torch.Tensor,
    model: SentenceTransformer,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Совместимая обертка над новыми функциями.
    """
    q_emb = encode_query(model, query)
    scores = compute_batch_scores(q_emb, corpus_embeddings)
    
    # Для совместимости возвращаем List[Tuple[str, float]]
    top_k_list = select_top_k([], scores, corpus_texts, top_k)
    return [(text, score) for score, text in sorted(top_k_list, reverse=True)]
