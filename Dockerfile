FROM python:3.11-slim

ARG EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/scTenifoldpy

COPY pyproject.toml README.md LICENSE ./
COPY scTenifold ./scTenifold

RUN python -m pip install --upgrade pip && \
    if [ -n "$EXTRAS" ]; then \
        python -m pip install ".[${EXTRAS}]"; \
    else \
        python -m pip install .; \
    fi

WORKDIR /workspace

CMD ["scTenifold", "--help"]
