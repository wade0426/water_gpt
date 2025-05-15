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


def get_date_range(start_date=date.today(), days=30):
    """
    獲取當前日期和30天後的日期。

    Args:
        start_date (str): 開始日期，格式為 "YYYY-MM-DD"，預設為當前日期。
        days (int): 要增加的天數，預設為30天。

    Returns:
        tuple: 包含當前日期和30天後的日期。
    """
    # 1. 獲取當前日期
    # today = date.today()
    today = start_date

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

    # 要退一層 注意檔案位置
    with open(os.path.join(FolderPath, f"../water_outage_notices.json"), "w", encoding="utf-8") as f:
        # 將 list 轉換為 json 字串 並保留換行和縮排
        f.write(json.dumps(response, ensure_ascii=False, indent=4))

    print(f"停水公告已保存至 {os.path.join(FolderPath, f'../water_outage_notices.json')}")


def find_matching_outages(data_list, affectedCounties, affectedTowns=None):
    """
    根據受影響的縣市和可選的鄉鎮市區篩選停水/降壓資料。

    Args:
        data_list (list): 包含停水/降壓事件的字典列表。
        affectedCounties (str or list/tuple/set of str): 
            要篩選的縣市代碼。必須是非空字串或非空字串的非空集合。
        affectedTowns (str or list/tuple/set of str, optional): 
            要篩選的鄉鎮市區代碼。預設為 None (不按鄉鎮市區篩選)。
            如果提供空字串 "" 或空集合 []/{}/()，則會匹配 'affectedTowns' 欄位也為空的項目。
            如果提供非空字串或非空集合，則所有元素必須是非空字串。

    Returns:
        list: 一個新的列表，其中只包含符合條件的字典。
    
    Raises:
        TypeError: 如果篩選條件的輸入類型不正確。
        ValueError: 如果 `affectedCounties` 為空或包含無效元素，
                    或者 `affectedTowns` (如果提供) 包含無效元素。
    """
    filtered_results = []

    # --- 驗證和標準化 affectedCounties 篩選條件 ---
    target_counties_set = set()
    if isinstance(affectedCounties, str):
        if not affectedCounties:  # 空字串
            raise ValueError("`affectedCounties` 字串篩選條件不可為空。")
        target_counties_set = {affectedCounties}
    elif isinstance(affectedCounties, (list, tuple, set)):
        if not affectedCounties:  # 空集合
            raise ValueError("`affectedCounties` 集合篩選條件不可為空。")
        # 檢查集合內是否有非字串或空字串元素
        for c in affectedCounties:
            if not isinstance(c, str) or not c:
                raise ValueError("`affectedCounties` 集合中的所有元素都必須是非空字串。")
        target_counties_set = set(affectedCounties)
    else:
        raise TypeError(
            "`affectedCounties` 必須是非空字串或非空字串的非空集合 (list, tuple, set)。"
        )

    # --- 驗證和標準化 affectedTowns 篩選條件 (如果提供) ---
    target_towns_set = None  # 預設：不按鄉鎮市區篩選
    if affectedTowns is not None:
        if isinstance(affectedTowns, str):
            # 如果鄉鎮市區為空字串，表示 "匹配沒有鄉鎮市區的項目"
            # 因此，target_towns_set 變成一個空集合
            target_towns_set = {affectedTowns} if affectedTowns else set()
        elif isinstance(affectedTowns, (list, tuple, set)):
            # 如果 affectedTowns 是空列表/元組/集合，target_towns_set 變成空集合。
            # 這表示 "僅匹配 'affectedTowns' 欄位也為空的項目"。
            # 僅當集合非空時才驗證其元素
            if affectedTowns: 
                for t in affectedTowns: 
                    if not isinstance(t, str) or not t:
                        raise ValueError("非空的 `affectedTowns` 集合中的所有元素都必須是非空字串。")
            target_towns_set = set(affectedTowns)
        else:
            raise TypeError(
                "`affectedTowns` 必須是字串、字串集合 (list, tuple, set) 或 None。"
            )

    # --- 迭代並篩選資料 ---
    for item in data_list:
        item_actual_counties = set(item.get('affectedCounties', []))
        
        # 1. 縣市匹配:
        # target_counties_set 和 item_actual_counties 之間必須有交集。
        if not (target_counties_set & item_actual_counties):
            continue  # 沒有匹配的縣市

        # 2. 鄉鎮市區匹配 (如果鄉鎮市區篩選條件啟用):
        if target_towns_set is not None:
            item_actual_towns = set(item.get('affectedTowns', []))
            
            # 不匹配的情況發生於:
            # A) target_towns_set 為空 (篩選條件是 "沒有鄉鎮市區") 且 item_actual_towns 非空。
            # 或
            # B) target_towns_set 非空 (篩選條件是特定的鄉鎮市區) 且兩者之間沒有交集。
            if (not target_towns_set and item_actual_towns) or \
               (target_towns_set and not (target_towns_set & item_actual_towns)):
                continue # 沒有匹配的鄉鎮市區

        # 如果所有條件都通過
        filtered_results.append(item)
            
    return filtered_results


if __name__ == "__main__":
    # 取得各縣市行政區資料
    # get_town_data()

    # 取得停水公告
    # get_water_outage_notices()

    with open(os.path.join(FolderPath, f"../water_outage_notices.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    affectedCounties = ["66000","10020"]
    results = find_matching_outages(data, affectedCounties=affectedCounties)
    print(len(results))

    pass