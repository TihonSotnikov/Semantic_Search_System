FROM ghcr.io/astral-sh/uv:debian
WORKDIR /home/sss
COPY . .
RUN uv sync
ENTRYPOINT [ "uv", "run", "app/main.py" ]
CMD [ "--host", "0.0.0.0", "--port", "8000" ]