import requests
from bs4 import BeautifulSoup
import json

headers = {
    'accept': '*/*',
    'accept-language': 'zh-TW,zh;q=0.9',
    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'origin': 'https://www.water.gov.tw',
    'priority': 'u=1, i',
    'referer': 'https://www.water.gov.tw/ch/Location?nodeId=4890',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0',
    'x-requested-with': 'XMLHttpRequest',
}

data = {
    'X-Requested-With': 'XMLHttpRequest',
}


# 將HTML內容解析並輸出為JSON檔案
def html_to_json_file(html_content):
    """
    將HTML內容解析並輸出為JSON檔案
    
    Args:
        html_content (str): HTML內容字串
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {}
    
    # 根據標籤文字搜尋對應的值
    def get_text_by_label(label):
        # 尋找包含標籤文字的 <th> 元素
        th = soup.find('th', string=lambda x: x and label in x)
        if th:
            # 值通常在下一個相鄰的 <td> 元素中
            td = th.find_next_sibling('td')
            if td:
                return td.get_text(strip=True)
        return None
    
    # 提取各個欄位
    data['code'] = get_text_by_label('代 碼')
    data['jurisdiction'] = get_text_by_label('轄 區')
    data['area_description'] = get_text_by_label('轄區介紹')
    data['contact_person'] = get_text_by_label('聯 絡 人')
    data['phone'] = get_text_by_label('電 話')
    data['fax'] = get_text_by_label('傳 真')
    data['service_email'] = get_text_by_label('服務信箱')
    data['address'] = get_text_by_label('地 址')
    data['traffic_info'] = get_text_by_label('交通資訊')
    data['note'] = get_text_by_label('備 註')
    
    # 更新時間不在表格中，需要單獨搜尋包含「更新時間」的文字
    update_time_tag = soup.find(string=lambda text: text and '更新時間' in text)
    if update_time_tag:
        # 從字串中提取時間
        data['update_time'] = update_time_tag.strip().replace('更新時間：', '').strip()
    else:
        data['update_time'] = None
    
    return data


def url_to_json_file(url):
    """
    將URL轉換為JSON檔案
    
    Args:
        url (str): URL
    Returns:
        dict: JSON檔案
    """
    response = requests.get(url, headers=headers, data=data)
    return html_to_json_file(response.text)


def main():
    with open("./water_gpt/water_location_data_v1.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        url = item['href']
        if url:
            url_file_result = url_to_json_file(url)
            item['code'] = url_file_result['code']
            item['jurisdiction'] = url_file_result['jurisdiction']
            item['area_description'] = url_file_result['area_description']
            item['contact_person'] = url_file_result['contact_person']
            item['phone'] = url_file_result['phone']
            item['fax'] = url_file_result['fax']
            item['service_email'] = url_file_result['service_email']
            item['address'] = url_file_result['address']
            item['traffic_info'] = url_file_result['traffic_info']
            item['note'] = url_file_result['note']
            item['update_time'] = url_file_result['update_time']

    with open("./water_gpt/water_location_data_v2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    main()
    print("done")