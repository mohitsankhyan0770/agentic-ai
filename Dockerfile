FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Run as a non-root user; /data holds files the agent writes
RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

ENV AGENT_WORKSPACE=/data/workspace \
    AGENT_MEMORY_DIR=/data/memory \
    AGENT_TRACE_DIR=/data/runs

EXPOSE 8000
CMD ["uvicorn", "agentic_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]