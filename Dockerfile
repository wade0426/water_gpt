FROM python:3.10-slim

WORKDIR /app

# 只複製 requirements.txt 先安裝依賴
COPY embedding_requirements.txt /app/

# 升級 pip 並安裝依賴
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r embedding_requirements.txt

# 創建必要目錄
RUN mkdir -p /app/models /app/data /app/db

# 設置環境變數
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data/water_data_content_v3-class.json
ENV DB_DIR=/app/db
ENV WATCHFILES_FORCE_POLLING=true

EXPOSE 8003

# 使用 uvicorn 並啟用熱重載
CMD ["uvicorn", "embedding-api:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]

