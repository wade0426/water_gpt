from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opencc import OpenCC
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json
import os
import logging
import time
import asyncio
import shutil  # 添加缺失的 import
from typing import List  # 添加缺失的 import

app = FastAPI()

# 設置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppRequest(BaseModel):
    request: str

class Top_kRequest(BaseModel):
    top_k: int

class AppResponse(BaseModel):
    response: list

class EmbeddingRequest(BaseModel):
    request: str
    top_k: int = 5

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_heartbeat: dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.last_heartbeat[websocket] = time.time()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.last_heartbeat.pop(websocket, None)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def _ping(self, websocket: WebSocket):
        try:
            while True:
                await asyncio.sleep(20)  # 每 20 秒 ping 一次
                await websocket.send_text("__ping__")
        except Exception as e:
            logging.warning(f"Ping 發送失敗，視為斷線：{e}")
            await websocket.close()
            self.disconnect(websocket)

class Embedding:
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

        def __repr__(self):
            return f"Document({self.page_content[:20]!r}, meta={self.metadata})"

    def __init__(self):
        # 使用環境變數替代硬編碼路徑
        self.DATA_PATH = os.getenv("DATA_PATH", "./data/water_data_content_v3-class.json")
        self.DB_DIR = os.getenv("DB_DIR", "./db")
        self.EMB_MODEL_NAME = os.getenv("MODEL_PATH", "./models")
        
        # 設置模型參數
        self.EMB_MODEL_KWARGS = {"device": "cpu", "trust_remote_code": True}
        
        # 檢查是否有 GPU 可用
        try:
            import torch
            if torch.cuda.is_available():
                self.EMB_MODEL_KWARGS["device"] = "cuda"
                print("→ 使用 GPU 模式")
            else:
                print("→ 使用 CPU 模式")
        except ImportError:
            print("→ PyTorch 未安裝，使用 CPU 模式")

        # 確保目錄存在
        os.makedirs(self.DB_DIR, exist_ok=True)
        
        print(f"→ 數據文件路徑: {self.DATA_PATH}")
        print(f"→ 數據庫目錄: {self.DB_DIR}")
        print(f"→ 模型路徑: {self.EMB_MODEL_NAME}")

        embedding = HuggingFaceEmbeddings(
            model_name=self.EMB_MODEL_NAME,
            model_kwargs=self.EMB_MODEL_KWARGS
        )

        self.tw2s = OpenCC('tw2s')  # 繁轉簡
        self.s2tw = OpenCC('s2tw')  # 簡轉繁

        self.vectordb = self.build_or_load_vectordb(embedding)

    def retrieve(self, query: str, top_k=5):
        q_simp = query#self.tw2s.convert(query)
        docs = self.vectordb.similarity_search_with_score(
            query=q_simp,
            k=top_k
        )

        results = []
        for doc, score in docs:
            results.append({
                "title": doc.metadata.get("title"),
                "content": doc.page_content,#self.s2tw.convert(doc.page_content),
                "category": doc.metadata.get("category"),
                "confidence": float(score)
            })

        return results

    def build_or_load_vectordb(self, embedding):
        # 檢查是否有有效的向量庫
        if os.path.exists(self.DB_DIR) and os.listdir(self.DB_DIR):
            try:
                print("→ 載入已有向量庫")
                return Chroma(persist_directory=self.DB_DIR, embedding_function=embedding)
            except Exception as e:
                print(f"→ 載入向量庫失敗: {e}，重新建立向量庫")

        print("→ 向量庫不存在或無效，開始建庫 …")
    
        # 檢查數據文件是否存在
        if not os.path.exists(self.DATA_PATH):
            raise FileNotFoundError(f"數據文件不存在: {self.DATA_PATH}")
        
        with open(self.DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs = []
        for item in data:
            docs.append(
                self.Document(
                    page_content=item["page_content"],#self.tw2s.convert(item["page_content"]),
                    metadata={
                        "title": item["title"],
                        "category": str(item["category"])
                    }
                )
            )

        # 清空目錄內容而不是刪除整個目錄
        if os.path.exists(self.DB_DIR):
            for filename in os.listdir(self.DB_DIR):
                file_path = os.path.join(self.DB_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'→ 刪除 {file_path} 失敗: {e}')

        vectordb = Chroma.from_documents(
            docs,  # positional: documents
            embedding,  # positional: embeddings
            persist_directory=self.DB_DIR
        )

        print("✔ 向量庫建置完成")
        return vectordb


# 全局變數
connection_manager = ConnectionManager()
main = None

@app.on_event("startup")
async def startup_event():
    global main
    main = Embedding()

@app.websocket("/ws/embedding")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            request = data.get("request")
            top_k = data.get("top_k", 5)
            
            result = main.retrieve(request, top_k)
            await connection_manager.send_personal_message(
                json.dumps({"response": result}),
                websocket
            )
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            await connection_manager.send_personal_message(
                json.dumps({"error": str(e)}),
                websocket
            )
        except:
            pass
        connection_manager.disconnect(websocket)

@app.post("/embedding")
async def get_embedding(request: EmbeddingRequest):
    try:
        result = main.retrieve(request.request, request.top_k)
        return {"response": result}
    except Exception as e:
        logging.error(f"Error processing embedding request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8003)  # 改為 8003 以匹配 Docker 配置

