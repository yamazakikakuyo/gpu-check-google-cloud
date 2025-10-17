FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-4:latest

ENV PORT=8080 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /src

EXPOSE 8080

# WORKDIR /app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
# ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
ENTRYPOINT ["python", "serve_main.py"]

