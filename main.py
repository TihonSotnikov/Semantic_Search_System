"""
Основное приложение FastAPI для семантического поиска по корпоративной базе знаний.

Модуль обеспечивает загрузку ML-модели, подготовку корпуса текстов,
выполнение семантического поиска и предоставление веб-интерфейса.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

MODEL_NAME: str = "cointegrated/rubert-tiny2"
DATA_PATH: str = "data/data.json"
TOP_K: int = 3
HOST: str = "0.0.0.0"
PORT: int = 8000

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (КЭШ)
# ============================================================================

model: Optional[SentenceTransformer] = None
data: list[dict] = []
corpus_embeddings: Optional[torch.Tensor] = None

# ============================================================================
# ФУНКЦИИ ЗАГРУЗКИ И ПОДГОТОВКИ
# ============================================================================


def load_json_data(filepath: str) -> list[dict]:
    """
    Загружает данные из JSON-файла.

    Parameters
    ----------
    filepath : str
        Путь к JSON-файлу с документами.

    Returns
    -------
    list[dict]
        Список документов с полями 'id', 'title', 'text'.

    Raises
    ------
    FileNotFoundError
        Если файл не найден.
    json.JSONDecodeError
        Если файл не валиден JSON.

    """

    data_path = Path(filepath)
    if not data_path.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден")

    with open(data_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    logger.info(f"Загружено {len(loaded_data)} документов из {filepath}")
    return loaded_data


def load_ml_model(model_name: str) -> SentenceTransformer:
    """
    Загружает предобученную модель для векторизации текста.

    Parameters
    ----------
    model_name : str
        Идентификатор модели на HuggingFace Hub.

    Returns
    -------
    SentenceTransformer
        Инициализированная модель для кодирования текстов.

    """

    logger.info(f"Загрузка модели {model_name}...")
    loaded_model = SentenceTransformer(model_name)
    logger.info("Модель загружена успешно")
    return loaded_model


def prepare_corpus(
    documents: list[dict],
    loaded_model: SentenceTransformer
) -> torch.Tensor:
    """
    Подготавливает корпус текстов и вычисляет их векторные представления.

    Parameters
    ----------
    documents : list[dict]
        Список документов с полем 'text'.
    loaded_model : SentenceTransformer
        Загруженная модель для кодирования.

    Returns
    -------
    torch.Tensor
        Матрица эмбеддингов размера (num_docs, embedding_dim).

    """

    texts = [doc["text"] for doc in documents]
    logger.info(f"Векторизация {len(texts)} текстов...")
    embeddings = loaded_model.encode(texts, convert_to_tensor=True)
    logger.info("Векторизация завершена")
    return embeddings


def perform_search(
    query: str,
    documents: list[dict],
    loaded_model: SentenceTransformer,
    embeddings: torch.Tensor,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Выполняет семантический поиск по корпусу и возвращает релевантные документы.

    Parameters
    ----------
    query : str
        Текст поискового запроса.
    documents : list[dict]
        Исходные документы из базы знаний.
    loaded_model : SentenceTransformer
        Загруженная модель для кодирования.
    embeddings : torch.Tensor
        Предвычисленные эмбеддинги документов.
    top_k : int
        Количество возвращаемых результатов, по умолчанию TOP_K.

    Returns
    -------
    list[dict]
        Список документов с добавленным полем 'score' (косинусное сходство).

    """

    query_embedding = loaded_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]

    k = min(top_k, len(documents))
    top_results = torch.topk(cos_scores, k=k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        doc_copy = documents[idx.item()].copy()
        doc_copy["score"] = float(score.item())
        results.append(doc_copy)

    logger.info(f"Найдено {len(results)} результатов для запроса: '{query}'")
    return results


# ============================================================================
# HTML-ФОРМА
# ============================================================================


def load_html_template() -> str:
    """
    Возвращает HTML-код для веб-интерфейса поиска.

    Returns
    -------
    str
        HTML-контент страницы.

    """

    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Семантический поиск</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 40px;
            }

            h1 {
                margin-bottom: 30px;
                color: #333;
                text-align: center;
            }

            .search-box {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
            }

            #query {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                font-size: 16px;
                transition: border-color 0.3s;
            }

            #query:focus {
                outline: none;
                border-color: #667eea;
            }

            button {
                padding: 12px 24px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                cursor: pointer;
                transition: background 0.3s;
            }

            button:hover {
                background: #764ba2;
            }

            button:active {
                transform: scale(0.98);
            }

            .results {
                margin-top: 30px;
                min-height: 100px;
            }

            .result-item {
                padding: 16px;
                border: 1px solid #e0e0e0;
                margin-bottom: 12px;
                border-radius: 4px;
                background: #f9f9f9;
                transition: box-shadow 0.3s;
            }

            .result-item:hover {
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }

            .result-title {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 8px;
            }

            .result-text {
                color: #555;
                margin-bottom: 8px;
                line-height: 1.5;
            }

            .result-score {
                color: #999;
                font-size: 0.9em;
            }

            .loading {
                color: #667eea;
                font-style: italic;
            }

            .no-results {
                color: #999;
                text-align: center;
                padding: 20px;
            }

            .error {
                color: #d32f2f;
                text-align: center;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Поиск по базе знаний</h1>
            <div class="search-box">
                <input
                    type="text"
                    id="query"
                    placeholder="Введите запрос..."
                    autocomplete="off"
                >
                <button onclick="performSearch()">Поиск</button>
            </div>
            <div id="results" class="results"></div>
        </div>

        <script>
            async function performSearch() {
                const query = document.getElementById('query').value.trim();
                if (!query) return;

                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<p class="loading">Поиск...</p>';

                try {
                    const response = await fetch(
                        `/search?query=${encodeURIComponent(query)}`
                    );

                    if (!response.ok) {
                        resultsDiv.innerHTML =
                            '<p class="error">Ошибка сервера</p>';
                        return;
                    }

                    const responseData = await response.json();
                    const results = responseData.results;

                    if (results.length === 0) {
                        resultsDiv.innerHTML =
                            '<p class="no-results">Результатов не найдено</p>';
                        return;
                    }

                    let html = '';
                    for (const result of results) {
                        const percentage = (result.score * 100).toFixed(1);
                        html += `
                            <div class="result-item">
                                <div class="result-title">${
                                    escapeHtml(result.title)
                                }</div>
                                <div class="result-text">${
                                    escapeHtml(result.text)
                                }</div>
                                <div class="result-score">
                                    Совпадение: ${percentage}%
                                </div>
                            </div>
                        `;
                    }
                    resultsDiv.innerHTML = html;
                } catch (error) {
                    resultsDiv.innerHTML =
                        '<p class="error">Ошибка подключения</p>';
                    console.error('Search error:', error);
                }
            }

            function escapeHtml(text) {
                const map = {
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#039;'
                };
                return text.replace(/[&<>"']/g, m => map[m]);
            }

            // Поиск при нажатии Enter
            document.getElementById('query').addEventListener(
                'keypress',
                function(event) {
                    if (event.key === 'Enter') {
                        performSearch();
                    }
                }
            );
        </script>
    </body>
    </html>
    """

    return html.strip()


# ============================================================================
# СОЗДАНИЕ FASTAPI ПРИЛОЖЕНИЯ
# ============================================================================


def create_app(
    loaded_model: SentenceTransformer,
    loaded_data: list[dict],
    loaded_embeddings: torch.Tensor,
    html_content: str
) -> FastAPI:
    """
    Создает и настраивает FastAPI приложение с маршрутами.

    Parameters
    ----------
    loaded_model : SentenceTransformer
        Загруженная модель для кодирования.
    loaded_data : list[dict]
        Загруженные документы.
    loaded_embeddings : torch.Tensor
        Предвычисленные эмбеддинги.
    html_content : str
        HTML-контент главной страницы.

    Returns
    -------
    FastAPI
        Настроенное приложение со всеми маршрутами.

    """

    app = FastAPI(
        title="Semantic Search API",
        description="API для семантического поиска по базе корпоративных знаний"
    )

    @app.get("/", response_class=HTMLResponse)
    async def get_root() -> str:
        """
        Возвращает HTML-интерфейс поиска.

        Returns
        -------
        str
            HTML-контент страницы.

        """

        return html_content

    @app.get("/search")
    async def search_endpoint(query: str) -> dict:
        """
        Выполняет семантический поиск по запросу.

        Parameters
        ----------
        query : str
            Поисковый запрос.

        Returns
        -------
        dict
            JSON с полями 'query' и 'results'.

        Raises
        ------
        HTTPException
            Если запрос пустой.

        """

        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Поисковый запрос не может быть пустым"
            )

        results = perform_search(
            query,
            loaded_data,
            loaded_model,
            loaded_embeddings,
            top_k=TOP_K
        )

        return {"query": query, "results": results}

    return app


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================


def main() -> None:
    """
    Инициализирует приложение и запускает сервер.

    Этапы:
    1. Загрузка данных из JSON
    2. Загрузка ML-модели
    3. Подготовка корпуса (векторизация)
    4. Создание FastAPI приложения
    5. Запуск Uvicorn сервера

    """

    global model, data, corpus_embeddings

    logger.info("=" * 70)
    logger.info("Инициализация приложения семантического поиска...")
    logger.info("=" * 70)

    try:
        # Загружаем данные
        data = load_json_data(DATA_PATH)

        # Загружаем модель
        model = load_ml_model(MODEL_NAME)

        # Подготавливаем корпус
        corpus_embeddings = prepare_corpus(data, model)

        # Загружаем HTML
        html_content = load_html_template()

        # Создаем приложение
        app = create_app(model, data, corpus_embeddings, html_content)

        # Запускаем сервер
        logger.info("=" * 70)
        logger.info(f"Запуск сервера на http://{HOST}:{PORT}")
        logger.info("Нажмите Ctrl+C для остановки")
        logger.info("=" * 70)

        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info"
        )

    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}")
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
