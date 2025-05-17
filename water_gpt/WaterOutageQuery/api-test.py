URL = "http://localhost:8002/water-outage-query"

import requests
import json

response = requests.get(URL, params={"affectedCounties": "66000", "affectedTowns": "66000080"})
# response = requests.get(URL, params={"affectedCounties": "66000"})
print(response.json())

# with open("response_test.json", "w", encoding="utf-8") as f:
#     json.dump(response.json(), f, ensure_ascii=False, indent=4)