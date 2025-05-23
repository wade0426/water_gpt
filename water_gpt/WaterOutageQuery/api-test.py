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