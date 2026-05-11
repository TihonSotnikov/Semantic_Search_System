from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


router = APIRouter(tags=['HTML'])
templates = Jinja2Templates('frontend/templates')


@router.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, 'index.html', {})

@router.get('/dashboard', response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, 'dashboard.html', {})
