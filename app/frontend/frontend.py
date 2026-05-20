import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


router = APIRouter(tags=['HTML'])
templates = Jinja2Templates(
    os.path.join(os.path.dirname(__file__), 'templates')
)


@router.get('/', response_class=HTMLResponse)
def index(request: Request):
    """
    Главная страница поисковой системы.

    Returns
    -------
    HTMLResponse
        HTML страница.
    """
    return templates.TemplateResponse(request, 'index.html', {})

@router.get('/dashboard', response_class=HTMLResponse)
def dashboard(request: Request):
    """
    Админ-панель сервиса.

    Returns
    -------
    HTMLResponse
        HTML страница
    """
    return templates.TemplateResponse(request, 'dashboard.html', {})
