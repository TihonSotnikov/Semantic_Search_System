# 🔍 Система семантического поиска по корпоративной базе знаний

## 📝 Описание проекта
Данный веб-сервис реализует технологию семантического поиска в базе данных при помощи NLP (Natural Language Processing) и векторного представления данных.

## 🚧 Проект находится в стадии разработки

## ⚙️ Стек технологий
* **Язык:** Python 3.10+
* **ML / NLP:** `sentence-transformers` (модель: `cointegrated/rubert-tiny2`)
* **Бэкенд:** Uvicorn + FastAPI
* **База данных:** SQLite (json-файлы для ранней разработки)
* **Фронтенд:** HTML / CSS / JavaScript

## 🚀 Установка и запуск

### Менеджер окружений
1. Установка `uv`: [Инструкция](https://docs.astral.sh/uv/getting-started/installation/)
2. Подключение виртуального окружения и скачивание зависимостей:
```sh
uv sync
```
3. Запуск:
```sh
uv run uvicorn main:app --port 8000
```

### Legacy
1. Создание виртуального окружения:
```bash
python -m venv .venv
```
2. Активация окружения:
* Windows: `.venv\Scripts\activate`
* macOS/Linux: `source .venv/bin/activate`
3. Установка зависимостей:
```bash
pip install -r requirements.txt
```

## 📁 Структура проекта
```text
Semantic_Search_System/
│
├── data/                   # Синтетические данные (документы для загрузки)
├── src/                    
│   ├── ml/
|   |   └── ml_engine.py    # Логика векторизации и косинусного сходства
|   |
|   └── database/
|       └── database.py     # Сохранение и чтение документов
|
├── frontend/               # Файлы веб-интерфейса (index.html, style.css)
├── docs
│   ├── plan.md             # План разработки и взаимодействия
│   └── responsibilities.md # Распределение обязанностей
├── main.py                 # Основной файл FastAPI (эндпоинты)
├── requirements.txt        # Список зависимостей Python
└── README.md               # Документация проекта
```