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
            self.metadata     = metadata

        def __repr__(self):
            return f"Document({self.page_content[:20]!r}, meta={self.metadata})"

    def __init__(self):
        self.DATA_PATH = "../water_data_content_v3-class.json"
        self.DB_DIR    = "./db"
        self.EMB_MODEL_NAME   = "C:/Users/ramune/Documents/Project/Python/rag/jina-embeddings-v3"
        self.EMB_MODEL_KWARGS = {"device": "cuda", "trust_remote_code": True}
        
        embedding = HuggingFaceEmbeddings(
            model_name=self.EMB_MODEL_NAME,
            model_kwargs=self.EMB_MODEL_KWARGS
        )
        
        self.tw2s   = OpenCC('tw2s')  # 繁轉簡
        self.s2tw = OpenCC('s2tw')  # 簡轉繁

        #self.top_k = 5

        #self.CATEGORY_MAP = {
        #    1: "電子帳單、簡訊帳單及通知服務",
        #    2: "帳單與繳費管理",
        #    3: "用戶帳戶與用水設備管理",
        #    4: "水質、淨水與生活應用",
        #    5: "污水下水道與污水使用費",
        #    6: "緊急停水、計畫停水與應變",
        #    7: "水價政策與事業經營",
        #    8: "App／網站使用與隱私政策",
        #}
        
        self.vectordb = self.build_or_load_vectordb(embedding)

    def retrieve(self, query: str, top_k=5):
        q_simp = self.tw2s.convert(query)
        docs = self.vectordb.similarity_search_with_score(
            query=q_simp,
            k=top_k#self.top_k
        )

        results = []
        for doc, score in docs:
            results.append({
                "title":      doc.metadata.get("title"),
                "content":    self.s2tw.convert(doc.page_content),
                "category":   doc.metadata.get("category"),
                # confidence 取原始 cosine score，或可以做归一化
                "confidence": float(score)
            })

        #results = [
        #    {
        #        "title":    d.metadata["title"],
        #        "content":  self.s2tw.convert(d.page_content),
        #        "category": d.metadata["category"],
        #        "score":    None
        #    }
        #    for d in docs
        #]

        return results

    def build_or_load_vectordb(self, embedding):
        if os.path.exists(self.DB_DIR):
            print("→ 載入已有向量庫")
            return Chroma(persist_directory=self.DB_DIR, embedding_function=embedding)

        print("→ 向量庫不存在，開始建庫 …")
        with open(self.DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs = []
        for item in data:
            docs.append(
                self.Document(
                    page_content=self.tw2s.convert(item["page_content"]),
                    metadata={
                        "title":    item["title"],
                        "category": str(item["category"])
                    }
                )
            )

        # 重建資料夾
        if os.path.exists(self.DB_DIR):
            shutil.rmtree(self.DB_DIR)

        vectordb = Chroma.from_documents(
            docs,                   # positional: documents
            embedding,              # positional: embeddings
            persist_directory=self.DB_DIR
        )
        print("✔ 向量庫建置完成")
        return vectordb

if __name__ == "__main__":
    import uvicorn
    connection_manager = ConnectionManager()
    main = Embedding()
    
    @app.websocket("/ws/embedding")
    async def websocket_endpoint(websocket: WebSocket):
        await connection_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                request = data.get("request")
                
                if data.get("top_k"):
                    top_k = data.get("top_k")
                else:
                    top_k = 5

                result = main.retrieve(request, top_k)#"True"

                await connection_manager.send_personal_message(
                    json.dumps({"response": result}),
                    websocket
                )
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            await connection_manager.send_personal_message(
                json.dumps({"error": str(e)}),
                websocket
            )
            connection_manager.disconnect(websocket)

    @app.post("/embedding")
    async def get_embedding(request: EmbeddingRequest):
        try:
            result = main.retrieve(request.request, request.top_k)
            return {"response": result}
        except Exception as e:
            logging.error(f"Error processing embedding request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    uvicorn.run(app, host="0.0.0.0", port=8001)