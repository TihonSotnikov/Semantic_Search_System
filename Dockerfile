FROM ghcr.io/astral-sh/uv:debian
WORKDIR /home/sss/app
COPY . ..
RUN uv sync
ENTRYPOINT [ "uv", "run", "uvicorn", "main:app" ]
CMD [ "--host", "0.0.0.0", "--port", "8000" ]