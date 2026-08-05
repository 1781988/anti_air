FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY anti_air ./anti_air
COPY configs ./configs
COPY scripts ./scripts
COPY infer.py train.py evaluate.py extract_features.py run.sh ./
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "infer.py"]
