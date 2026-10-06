FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-xgboost.txt ./
RUN pip install --no-cache-dir -r requirements-xgboost.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e . && useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
