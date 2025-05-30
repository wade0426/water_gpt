URL = "http://localhost:8002/water-outage-query"
URL_LOCATION = "http://localhost:8002/water-location-query"

import requests
import json

#response = requests.get(URL, params={"affectedCounties": "66000", "affectedTowns": "66000050", "query": "code"})
# response = requests.get(URL, params={"affectedCounties": "臺中市", "affectedTowns": "北區", "query": "name"})

#response = requests.get(URL, params={"affectedCounties": "66000"})
# response = requests.get(URL, params={"affectedCounties": "臺中市", "query": "name"})
# response = requests.get(URL, params={"affectedCounties": "臺中市", 
#                                      "query": "name", 
#                                      "startDate": "2025-06-01", 
#                                      "endDate": "2025-06-02"})

# 測試地址關鍵字
#response = requests.get(URL, params={"affectedCounties": "臺中市", 
#                                     "query": "name", 
#                                     "startDate": "2025-05-01", 
#                                     "endDate": "2025-06-30",
#                                     "addressKeyword": "三民路三段"})

# print(len(response.json()["result"]))
#print(response.json())

# with open("response_test.json", "w", encoding="utf-8") as f:
#     json.dump(response.json(), f, ensure_ascii=False, indent=4)


response = requests.get(URL_LOCATION, params={"location": "埔里鎮"})
print(response.json())