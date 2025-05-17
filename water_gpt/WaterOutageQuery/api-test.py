URL = "http://localhost:8002/water-outage-query"

import requests

#  參數 affectedCounties = ["66000"]
#  參數 affectedTowns = ["10020"]

# response = requests.get(URL, params={"affectedCounties": "66000", "affectedTowns": "10020"})
response = requests.get(URL, params={"affectedCounties": "66000"})
print(response.json())

