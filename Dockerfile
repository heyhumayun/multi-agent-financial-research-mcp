FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FIN_RESEARCH_TOOL_RUNTIME=mcp-stdio

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[api,market]"

EXPOSE 8000

CMD ["financial-research-api"]
