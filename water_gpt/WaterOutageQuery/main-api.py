from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Tools import *
import json
from datetime import datetime, timedelta
import os

# 資料夾路徑
FolderPath = "./"

app = FastAPI()

# 設置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_county_district_dict():
    # 讀取 GetCounty.json 取得所有縣市的 key
    with open(os.path.join(FolderPath, "County_data/GetCounty.json"), "r", encoding="utf-8") as f:
        counties = json.load(f)

    # 準備結果 dict
    counties_dict = {}

    towns_dict = {}

    # counties 格式假設為 [{"label": "新北市", "value": "65000"}, ...]
    for county in counties:
        county_key = county['value']
        county_file = os.path.join("County_data", f"{county_key}.json")
        # 檢查檔案是否存在
        if not os.path.exists(county_file):
            print(f"檔案不存在: {county_file}，跳過")
            continue

        # 讀取每個縣市的 json
        with open(county_file, "r", encoding="utf-8") as f:
            districts = json.load(f)

        # districts 假設為 [{"label": "五股區", "value": "65000150"}, ...]
        district_dict = {d["label"]: d["value"] for d in districts}
        towns_dict[county_key] = district_dict

    counties_dict = {county["label"]: county["value"] for county in counties}
    return counties_dict, towns_dict

all_counties_dict, all_towns_dict = build_county_district_dict()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/water-outage-query")
async def water_outage_query(affectedCounties: str, affectedTowns: str = None, query: str = 'code'):
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

    if query == 'code':
        result = find_matching_outages(data, affectedCounties, affectedTowns)
    elif query == 'name':
        county_value = all_counties_dict[affectedCounties]
        town_value = all_towns_dict[county_value][affectedTowns]
        result = find_matching_outages(data, county_value, town_value)

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