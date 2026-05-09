FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install third-party deps only — cached as long as pyproject.toml doesn't change.
# --no-install-project skips building the local package (src/ not copied yet).
# CPU-only torch index avoids pulling 2 GB of CUDA binaries that are useless here.
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && uv cache clean

# Copy source and install the local package (fast — deps already cached above).
COPY . .
RUN uv sync --no-dev \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && uv cache clean

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
