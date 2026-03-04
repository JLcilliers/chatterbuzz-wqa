FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY packages/api/requirements.txt ./api-requirements.txt
COPY packages/wqa-engine/requirements.txt ./wqa-requirements.txt
COPY packages/data-pipeline/requirements.txt ./pipeline-requirements.txt
COPY packages/content-engine/requirements.txt ./content-requirements.txt
COPY packages/schema-generator/requirements.txt ./schema-requirements.txt

RUN pip install --no-cache-dir \
    -r api-requirements.txt \
    -r wqa-requirements.txt \
    -r pipeline-requirements.txt \
    -r content-requirements.txt \
    -r schema-requirements.txt

# Copy all Python packages
COPY packages/ ./packages/

# Create symlinks so each package's src is importable by a clean name
RUN ln -s /app/packages/wqa-engine/src /app/wqa_engine && \
    ln -s /app/packages/data-pipeline/src /app/data_pipeline && \
    ln -s /app/packages/content-engine/src /app/content_engine && \
    ln -s /app/packages/schema-generator/src /app/schema_generator && \
    ln -s /app/packages/api /app/api_pkg

ENV PYTHONPATH="/app:/app/packages/api"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
