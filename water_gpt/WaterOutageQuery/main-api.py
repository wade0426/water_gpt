from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Tools import *
import json
from datetime import datetime, timedelta
import os

# 資料夾路徑
FolderPath = "D:/Python/NLP_LAB/water/water_gpt/water_gpt/WaterOutageQuery"

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

    # 檢查最後更新時間
    if os.path.exists(os.path.join(FolderPath, "last_update.txt")):
        with open(os.path.join(FolderPath, "last_update.txt"), "r", encoding="utf-8") as f:
            last_update_time = f.read()
    else:
        # 沒有最後更新時間，自動建立。
        print(f"沒有最後更新時間，自動建立。")
        last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(os.path.join(FolderPath, "last_update.txt"), "w", encoding="utf-8") as f:
            f.write(last_update_time)
        get_water_outage_notices()

    # 檢查最後更新時間是否超過 5 分鐘
    if datetime.now() - datetime.strptime(last_update_time, "%Y-%m-%d %H:%M") > timedelta(minutes=5):
        print(f"最後更新時間已超過 5 分鐘，自動更新資料。")
        last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(os.path.join(FolderPath, "last_update.txt"), "w", encoding="utf-8") as f:
            f.write(last_update_time)
        get_water_outage_notices()

    with open(os.path.join(FolderPath, "water_outage_notices.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    result = find_matching_outages(data, affectedCounties, affectedTowns)

    # 定義要取得的欄位
    # waterOffNumber: 停水影響戶數
    # pressureDownNumber: 水壓降低影響戶數
    fields = ["no", "isSchedule", "startDate", "endDate", "startTime", "endTime", "waterOffRegion", "waterOffReason", "waterOffNumber", "pressureDownRegion", "pressureDownReason", "pressureDownNumber", "lastUpdatedTime", "contact", "note", "affectedCounties", "affectedTowns", "actualEndTime", "keywords", "removeReason"]
    # 篩選資料
    filtered_results = []
    for item in result:
        filtered_item = {field: item[field] for field in fields}
        filtered_results.append(filtered_item)
    

    return {"message": "success", "result": filtered_results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)