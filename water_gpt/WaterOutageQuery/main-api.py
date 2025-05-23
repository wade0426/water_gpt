from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Tools import *
import json
from datetime import datetime, timedelta
import os
import threading
import time

# 資料夾路徑
FolderPath = "./"

# 全局變量用於存儲停水資料
water_outage_data = []

# 讀取停水資料的函數
def load_water_outage_data():
    global water_outage_data
    try:
        with open(os.path.join(FolderPath, "water_outage_notices.json"), "r", encoding="utf-8") as f:
            water_outage_data = json.load(f)
    except FileNotFoundError:
        print("停水資料檔案不存在，將進行首次下載")
        get_water_outage_notices()
        with open(os.path.join(FolderPath, "water_outage_notices.json"), "r", encoding="utf-8") as f:
            water_outage_data = json.load(f)

# 檢查並更新停水資料的函數
def update_water_outage_data():
    while True:
        try:
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
                load_water_outage_data()

            # 檢查最後更新時間是否超過 5 分鐘
            if datetime.now() - datetime.strptime(last_update_time, "%Y-%m-%d %H:%M") > timedelta(minutes=5):
                print(f"最後更新時間已超過 5 分鐘，自動更新資料。")
                last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                with open(os.path.join(FolderPath, "last_update.txt"), "w", encoding="utf-8") as f:
                    f.write(last_update_time)
                get_water_outage_notices()
                load_water_outage_data()
            
            # 每分鐘檢查一次
            time.sleep(60)
        except Exception as e:
            print(f"更新停水資料時發生錯誤: {e}")
            time.sleep(60)  # 發生錯誤時，等待一分鐘後重試

# 初始化，載入停水資料
load_water_outage_data()

# 啟動更新線程
update_thread = threading.Thread(target=update_water_outage_data, daemon=True)
update_thread.start()

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

def find_full_name_and_code(keyword: str, data: dict) -> tuple[str, str] | None:
    for name in data:
        if name.startswith(keyword):
            return name, data[name]
    return None

def find_city_and_town_code(keyword: str, all_towns_dict: dict) -> tuple[str, str] | None:
    for city_code, towns in all_towns_dict.items():
        for town_name, town_code in towns.items():
            if keyword in town_name:
                return city_code, town_name
    return None

all_counties_dict, all_towns_dict = build_county_district_dict()

print(f"all_counties_dict: {all_counties_dict}")
print(f"all_towns_dict: {all_towns_dict}")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/water-outage-query")
async def water_outage_query(affectedCounties: str, affectedTowns: str = None, query: str = 'code'):
    try:
        # 使用全局變量就不需要每次重新讀取
        if query == 'code':
            result = find_matching_outages(water_outage_data, affectedCounties, affectedTowns)
        elif query == 'name':
            #affectedCounties = affectedCounties.replace("台", "臺")
            #affectedTowns = affectedTowns.replace("台", "臺") if affectedTowns else None

            #result_counties = find_full_name_and_code(affectedCounties, all_counties_dict) #模糊查詢

            #if not result_counties:
                #county_value = all_counties_dict[affectedCounties]
            #if result_counties: # 直接找到對應的縣市
            #    print(result_counties)
            #    affectedCounties = result_counties[0]
#
            #if affectedTowns != None:
            #    result_towns = find_city_and_town_code(affectedTowns, all_towns_dict)
            #
            #if result_towns:
                
            #if result_towns: # 直接找到對應的鄉鎮市區
                #print(f'town:{result_towns}')

            #print(result_counties[0])  # 輸出 ('臺南市', '67000')

            county_value = all_counties_dict[affectedCounties]

            if affectedTowns is None:
                town_value = None
            else:
                town_value = all_towns_dict[county_value][affectedTowns]


            #print(f"county_value: {county_value}, town_value: {town_value}")
            
            result = find_matching_outages(water_outage_data, county_value, town_value)

        # 定義要取得的欄位
        # waterOffNumber: 停水影響戶數
        # pressureDownNumber: 水壓降低影響戶數
        fields = ["no", "isSchedule", "startDate", "endDate", "startTime", "endTime", "waterOffRegion", "waterOffReason", "waterOffNumber", "pressureDownRegion", "pressureDownReason", "pressureDownNumber", "contact", "note", "affectedCounties", "affectedTowns", "actualEndTime", "keywords", "removeReason"]
        # 篩選資料
        filtered_results = []
        for item in result:
            filtered_item = {field: item[field] for field in fields}
            filtered_results.append(filtered_item)
        

        return {"message": "success", "result": filtered_results}
    except Exception as e:
        print(f"發生錯誤: {e}")
        return {"message": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)