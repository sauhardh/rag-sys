FROM python:3.13-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv 

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]

