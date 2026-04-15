import torch
from sentence_transformers import SentenceTransformer, util
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

    return SentenceTransformer(model_name)


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


def search_similar_texts(
    query: str,
    corpus_texts: List[str],
    corpus_embeddings: torch.Tensor,
    model: SentenceTransformer,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Выполняет семантический поиск запроса по векторизованному корпусу.

    Parameters
    ----------
    query : str
        Текст поискового запроса.
    corpus_texts : List[str]
        Исходный массив текстов базы знаний.
    corpus_embeddings : torch.Tensor
        Предрассчитанные векторы базы знаний.
    model : SentenceTransformer
        Загруженная модель векторизации.
    top_k : int
        Количество возвращаемых релевантных результатов (по умолчанию 3).

    Returns
    -------
    List[Tuple[str, float]]
        Список результатов в формате (текст_документа, оценка_косинусного_сходства).
    """

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    k = min(top_k, len(corpus_texts))
    top_results = torch.topk(cos_scores, k=k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((corpus_texts[idx.item()], score.item()))

    return results
