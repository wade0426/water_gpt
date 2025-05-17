from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Tools import *
import json

OUTAGE_DATA = "D:/Python/NLP_LAB/water/water_gpt/water_gpt/WaterOutageQuery/water_outage_notices.json"

app = FastAPI()

# 設置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/water-outage-query")
async def water_outage_query(affectedCounties: str, affectedTowns: str = None):
    # 不需要從 request 物件獲取參數，FastAPI 會自動處理

    with open(OUTAGE_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = find_matching_outages(data, affectedCounties, affectedTowns)

    return {"message": "success", "result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)