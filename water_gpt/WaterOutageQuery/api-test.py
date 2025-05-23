URL = "http://localhost:8002/water-outage-query"

import requests
import json

#response = requests.get(URL, params={"affectedCounties": "66000", "affectedTowns": "66000050", "query": "code"})
# response = requests.get(URL, params={"affectedCounties": "臺中市", "affectedTowns": "北區", "query": "name"})

#response = requests.get(URL, params={"affectedCounties": "66000"})
# response = requests.get(URL, params={"affectedCounties": "臺中市", "query": "name"})
response = requests.get(URL, params={"affectedCounties": "臺中市", 
                                     "query": "name", 
                                     "startDate": "2025-06-01", 
                                     "endDate": "2025-06-02"})

# print(len(response.json()["result"]))
print(response.json())

# with open("response_test.json", "w", encoding="utf-8") as f:
#     json.dump(response.json(), f, ensure_ascii=False, indent=4)




def generate_water_off_notification(no, start_date, end_date, start_time, end_time, 
                                  water_off_region, water_off_reason, water_off_number, 
                                  contact, pressure_down_region=None, pressure_down_reason=None, 
                                  pressure_down_number=0, note=None):
    """
    生成停水資訊通知的markdown模板
    
    參數:
    - no: 編號
    - start_date: 開始日期 (格式: YYYY-MM-DD 或中文)
    - end_date: 結束日期 (格式: YYYY-MM-DD 或中文)
    - start_time: 開始時間 (格式: HH:MM 或中文)
    - end_time: 結束時間 (格式: HH:MM 或中文)
    - water_off_region: 停水區域
    - water_off_reason: 停水原因
    - water_off_number: 停水戶數
    - contact: 聯絡電話
    - pressure_down_region: 減壓區域 (可選)
    - pressure_down_reason: 減壓原因 (可選)
    - pressure_down_number: 減壓戶數 (可選)
    - note: 額外注意事項 (可選)
    """
    
    # 格式化日期時間
    if start_date and len(start_date) == 10 and start_date.count('-') == 2:
        start_date = start_date.replace('-', '年', 1).replace('-', '月') + '日'
    if end_date and len(end_date) == 10 and end_date.count('-') == 2:
        end_date = end_date.replace('-', '年', 1).replace('-', '月') + '日'
    
    if start_time and ':' in start_time:
        hour, minute = start_time.split(':')
        start_time = f"上午{hour}:{minute}" if int(hour) < 12 else f"下午{int(hour)-12 if int(hour) > 12 else hour}:{minute}"
    
    if end_time and ':' in end_time:
        hour, minute = end_time.split(':')
        end_time = f"上午{hour}:{minute}" if int(hour) < 12 else f"下午{int(hour)-12 if int(hour) > 12 else hour}:{minute}"
    
    template = f"""# 🚰 停水資訊通知

## 停水通知（編號：{no}）

### 📅 停水時間
- **日期**：{start_date} 至 {end_date}
- **時間**：{start_time} 至 {end_time}

### 📍 影響區域
{water_off_region}

### 🔧 停水原因
{water_off_reason}

### 📊 影響戶數
**{water_off_number:,}戶**"""

    # 如果有減壓資訊，加入減壓部分
    if pressure_down_region and pressure_down_number > 0:
        template += f"""

### ⚡ 減壓影響
- **減壓區域**：{pressure_down_region}
- **減壓原因**：{pressure_down_reason}
- **減壓戶數**：**{pressure_down_number:,}戶**"""

    template += f"""

### ☎️ 聯絡電話
**{contact}**

---

## ⚠️ 重要注意事項

1. **儲水準備**：停水範圍內用戶請自行儲水備用
2. **安全提醒**：停水期間請慎防火源，關閉抽水機電源
3. **防污染措施**：建築物自來水進水口低於地面的用戶，請關閉總表前制水閥
4. **復水時間**：管線末端及高地區域可能延遲復水
5. **進度查詢**：可至[停水查詢系統](https://web.water.gov.tw/wateroffmap/map)查詢停復水進度"""

    # 如果有額外注意事項，加入自定義note
    if note:
        template += f"""

## 📋 額外注意事項
{note}"""

    return template

# 使用範例1：基本停水通知
example1 = generate_water_off_notification(
    no='202505220001',
    start_date='2025-06-02',
    end_date='2025-06-06', 
    start_time='08:30',
    end_time='18:30',
    water_off_region='臺中市北區：中清路一段、太原路一段、忠太東路、忠明七街、忠明八街',
    water_off_reason='辦理北區五常街等汰換管線工程(忠太東路工區)路口改接(第一階段)',
    water_off_number=1638,
    contact='1910'
)

# 使用範例2：包含減壓資訊的通知
example2 = generate_water_off_notification(
    no='202505200059',
    start_date='2025-05-29',
    end_date='2025-06-05',
    start_time='08:30', 
    end_time='18:30',
    water_off_region='太平區：建興里、成功里、建國里、平安里、中平里',
    water_off_reason='辦理太平區正誠街等汰換管線工程(二)',
    water_off_number=4825,
    contact='04-22442469',
    pressure_down_region='太平區：建興里、成功里、建國里、平安里、中平里',
    pressure_down_reason='管線工程導致水壓降低，部分區域可能有無水情形',
    pressure_down_number=116
)

print("範例1輸出：")
print(example1)
print("\n" + "="*50 + "\n")
print("範例2輸出：")
print(example2)
