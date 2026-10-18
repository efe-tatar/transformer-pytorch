# CPU image for the API in serve/app.py. The model is downloaded from the Hub at build time,
# so the container starts without network access.
#
#   docker build -t base-100m .
#   docker run -p 8000:8000 base-100m
#   curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
#        -d '{"prompt": "Photosynthesis is the process by which", "max_new_tokens": 60}'
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/hf_cache

# CPU-only torch wheel: ~200 MB instead of ~2 GB with CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt

ARG MODEL_ID=Efe-Tatar/base-100m-fineweb-edu
ENV MODEL_ID=${MODEL_ID}
RUN python -c "from huggingface_hub import snapshot_download as d; d('${MODEL_ID}', allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json'])"

COPY config.py ./
COPY network network
COPY tokenizer tokenizer
COPY tools/__init__.py tools/export_hf.py tools/load_hf.py tools/
COPY serve serve

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
