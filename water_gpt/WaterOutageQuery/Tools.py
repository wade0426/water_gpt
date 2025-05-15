from datetime import date, timedelta
import json
import requests
import os


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0"
headers = {
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'accept-language': 'zh-TW,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'content-type': 'application/json',
    'origin': 'https://web.water.gov.tw',
    'priority': 'u=1, i',
    'referer': 'https://web.water.gov.tw/wateroff/city/%E8%87%BA%E4%B8%AD%E5%B8%82/index.html',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
    'x-requested-with': 'XMLHttpRequest',
}

# 注意替換
FolderPath = "D:/Python/NLP_LAB/water/water_gpt/water_gpt/WaterOutageQuery/County_data"


def get_date_range(days=30):
    """
    獲取當前日期和30天後的日期。

    Args:
        days (int): 要增加的天數，預設為30天。

    Returns:
        tuple: 包含當前日期和30天後的日期。
    """
    # 1. 獲取當前日期
    today = date.today()

    # 2. 將當前日期格式化為 "YYYY-MM-DD"
    formatted_today = today.strftime("%Y-%m-%d")

    # 3. 計算30天後的日期
    future_date_delta = timedelta(days=days)
    future_date = today + future_date_delta

    # 4. 將30天後的日期格式化為 "YYYY-MM-DD"
    formatted_future_date = future_date.strftime("%Y-%m-%d")

    # 5. 輸出結果
    # print(f"當前日期 {formatted_today}")
    # print(f"30天後的日期 {formatted_future_date}")
    return formatted_today, formatted_future_date


def remove_coordinates_from_water_off_area(data_list: list):
    """
    移除列表中每個項目中 'waterOffArea' 和 'pressureDownArea' 字典內的 'coordinates' 鍵。

    Args:
        data_list (list): 包含一個或多個字典的列表，
                          每個字典可能包含 'waterOffArea' 鍵
                          或 'pressureDownArea' 鍵。
    Returns:
        list: 修改後的列表。
    """
    for item in data_list:
        # 檢查 'waterOffArea' 是否存在且是一個字典
        if 'waterOffArea' in item and isinstance(item['waterOffArea'], dict):
            # 如果 'coordinates' 存在於 'waterOffArea' 中，則移除它
            # 使用 pop 並提供一個預設值 (None) 可以避免在鍵不存在時出錯
            item['waterOffArea'].pop('coordinates', None)
        if 'pressureDownArea' in item and isinstance(item['pressureDownArea'], dict):
            # 如果 'coordinates' 存在於 'pressureDownArea' 中，則移除它
            # 使用 pop 並提供一個預設值 (None) 可以避免在鍵不存在時出錯
            item['pressureDownArea'].pop('coordinates', None)
    return data_list


def remove_items_with_values(data_list: list):
    """
    移除列表中每個項目中 'value' = values_to_remove 的項目。

    Args:
        data_list (list): 包含一個或多個字典的列表，
                          每個字典可能包含 'value' 鍵。
    Returns:
        list: 修改後的列表。
    """
    values_to_remove = ['0', '10000'] # 要移除的 value 值 (是字串)
    return [item for item in data_list if item['value'] not in values_to_remove]


def extract_values_simple(data_list: list) -> list:
  """
  從一個包含字典的列表中，提取每個字典中 "value" 鍵對應的值。

  Args:
    data_list (list): 一個字典的列表，每個字典都包含 "value" 鍵。

  Returns:
    list: 包含所有 "value" 值的列表。
  """
  return [item["value"] for item in data_list if isinstance(item, dict) and "value" in item]


def get_town_data():
    """
    獲取所有縣市行政區資料
    """
    
    # 檢查 json 檔案 是否存在
    if not os.path.exists(os.path.join(FolderPath, "GetCounty.json")):
        print(f"GetCounty.json 不存在")
        return False

    # 讀檔，讀取縣市資料
    with open(os.path.join(FolderPath, "GetCounty.json"), "r", encoding="utf-8") as f:
        input_data = json.load(f)
    # 提取縣市資料(value)
    county_value_list = extract_values_simple(input_data)

    for GetTown_number in county_value_list:
        url = f"https://web.water.gov.tw/wateroffapi/disaster/GetTown/{GetTown_number}"
        response = requests.get(url, headers={"User-Agent": user_agent})
        data = json.loads(response.text)
        # 移除特定 value 值
        data = remove_items_with_values(data)

        # 存檔，指定 UTF-8 編碼
        with open(os.path.join(FolderPath, f"{GetTown_number}.json"), "w", encoding="utf-8") as f:
            # 將 list 轉換為 json 字串 並保留換行和縮排
            f.write(json.dumps(data, ensure_ascii=False, indent=4))


def get_water_outage_notices():
    """
    獲取停水公告
    """
    # 取得當前時間, 30天後的時間
    start_date, end_date = get_date_range()
    json_data = {
        'mode': 1,
        'startDate': start_date,
        'endDate': end_date,
    }
    response = requests.post('https://web.water.gov.tw/wateroffapi/f/case/search', headers=headers, json=json_data)
    # 移除 'waterOffArea' 字典內的 'coordinates' 鍵
    response = remove_coordinates_from_water_off_area(response.json())

    # 要退一層
    with open(os.path.join(FolderPath, f"../water_outage_notices.json"), "w", encoding="utf-8") as f:
        # 將 list 轉換為 json 字串 並保留換行和縮排
        f.write(json.dumps(response, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # get_date_range()
    # remove_items_with_values(data)
    # remove_coordinates_from_water_off_area(data)

    # get_town_data()

    get_water_outage_notices()

    pass