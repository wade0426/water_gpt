import requests
from langchain import PromptTemplate, LLMChain
import asyncio
import json
from langchain.llms.base import LLM
from datetime import datetime

API_URL = "http://4090p8000.huannago.com/v1/chat/completions"
WATER_OUTAGE_URL = "http://localhost:8002/water-outage-query"
EMBEDDING_URL = "http://3090p8001.huannago.com/embedding"
HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0"}
MODEL   = "gpt-3.5-turbo"


class ClassifierLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": "你是一個訊息分類器，只回覆單字 \"是\" 或 \"否\""},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @property
    def identifying_params(self) -> dict:
        return {"model": MODEL}

class StatusLLM(ClassifierLLM):  # 可繼承同樣底層
    def _call(self, prompt: str, stop=None) -> str:
        system_prompt = """你是一個狀態選擇器，負責根據輸入的對話紀錄判斷當前的意圖狀態。你的任務是分析對話內容，並從以下三種狀態中選擇一個作為當前意圖：
READY：正常對話，無特定意圖或未進入特定功能流程。
OUTAGE：查詢即時停水資訊，當用戶提及停水相關問題或查詢時進入此狀態。
RAG：一般 FAQ，當用戶提出常見問題或尋求一般資訊時進入此狀態。

規則：
狀態總是根據用戶最新一則訊息的意圖來判斷（不考慮助理當前流程），只要用戶問了新問題，就以新問題為準。
若用戶最後訊息是停水查詢（含「停水資訊」、「有沒有停水」、「查停水」等），則為 OUTAGE。
若用戶最後訊息是一般問題（如「如何繳水費」等非停水相關），則為 RAG。
若最後訊息無明確意圖（如寒暄），則為 READY。
若最後訊息意圖不明，則延續上一個明確狀態。

範例：
用戶最後詢問「如何繳水費？」→ {"status": "RAG"}
用戶最後詢問「這裡有停水嗎？」→ {"status": "OUTAGE"}
用戶最後說「你好」→ {"status": "READY"}

請根據以上規則分析以下對話紀錄並輸出結果。"""

        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

class RetrieveLLM(ClassifierLLM):  # 可繼承同樣底層
    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": "你是一個文件片段選擇器，只回覆文件片段所屬的編號"},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class EmotionLLM(ClassifierLLM):  # 可繼承同樣底層
    def _call(self, prompt: str, stop=None) -> str:
        system_prompt = """您是一個高度專業的情緒辨識雷達。情境是一位專業保險業務員的對話文本，您的任務是分析輸入的文本，請嚴格遵守以下指南:

1. 情緒類別定義:
  - 分析輸入的文本並歸類為以下5種情緒之一:anger, irritation, uncertainty, happiness, neutral
  **anger**文本帶有直覺且強烈的負面措辭:
    - 髒話或極度負面的措辭。
    - 針對性含意或惡言相向的措辭。
    - 侮辱性的人身攻擊措辭。

  **irritation**文本帶有輕度的負面含意:
    - 不耐煩、厭煩。
    - 挖苦、嘲諷、諷刺。
    - 委婉或間接的負面話語。
    - 絕對不會出現直覺的、明確指名的攻擊措辭。
    - 未達到"anger"的程度。

  **uncertainty**文本帶有不確定的態度:
    - 模糊不定、猶豫、疑惑。
    - 事件掌握程度不足。
    - 待確認。
    - 絕對不會出現詢問他人資訊或意見。

  **happiness**文本帶有快樂的態度:
    - 稱讚或讚美對方。
    - 感謝對方。
    - 認同對方觀點或行為。
    - 表達對事件的滿意與喜悅。

  **neutral**文本沒有明確的情緒傾向:
    - 陳述事實。
    - 敘事句。
    - 使用禮貌語言。
    - 語氣冷靜且平衡。
    - 詢問他人資訊或意見。

2. 關鍵詞匹配與語境規則：
  - uncertainty：關鍵詞包括「應該」、「不確定」、「不知道」、「查一下」、「好像」、「晚點再確認」。  
  - neutral：關鍵詞包括「最低投保金額」、「最高保額」、「我們的保費」。  
  - irritation：關鍵詞包括「之前」、「不能」、「好好記下來」、「剛剛有說了」，或諷刺性表達。  
  - anger：關鍵詞包括「笨」、「你很笨」、「給我滾」、「你很自私」。  
  - happiness：關鍵詞包括「很高興」、「很樂意」、「謝謝」、「很開心」、「您真專業」、「您真內行」、「您真聰明」。  

3. 注意事項:
   - 保持客觀，不要被文本的內容影響您的判斷。
   - 考慮文化和語境因素，但始終保持一致的分類標準。
   - 如遇到模棱兩可的情況，選擇最適合的單一標籤。
   - 若匹配到關鍵字，則歸納至對應情緒。
   - 肯定對方通常是happiness。

4. 輸出格式:
   - 僅輸出一個情緒標籤，不需要解釋或其他額外信息。
   - 確保輸出的標籤為小寫。

請根據以上指南,準確地將輸入文本歸類為5種情緒之一。"""

        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

class LocationOutageLLM(ClassifierLLM):
    def _call(self, prompt: str, stop=None) -> str:
        system_prompt = """你是一個地點捕捉器，判斷使用者輸入的內容所包含的地點。
【判斷條件】需同時符合：
1. 訊息中必須明確提及台灣地名（由使用者輸入直接提到），地名可分為：
   - Counties：一、二級行政區（如「臺中市」、「南投縣」）
   - Towns：三級行政區（如「北區」、「埔里鎮」）
   如果提及到多個一、二級行政區，則擷取首個。
   如果提及到多個三級行政區，則擷取首個。
   如果只提及三級行政區，則自動補全其所屬的一、二級行政區。
   如果提及三級行政區不在其所屬的一、二級行政區，則視為無效。
   地名必須在地名資料庫中存在，否則禁止輸出。

   - 地名資料庫：
[基隆市]
中正區, 中山區, 信義區, 仁愛區, 暖暖區, 安樂區, 七堵區

[新北市]
五股區, 八里區, 淡水區, 三芝區, 石門區, 林口區, 三重區, 蘆洲區, 金山區, 汐止區, 萬里區, 三峽區, 鶯歌區, 瑞芳區, 石碇區, 平溪區, 雙溪區, 貢寮區, 坪林區, 中和區, 永和區, 板橋區, 樹林區, 新莊區, 新店區, 土城區, 烏來區, 泰山區, 深坑區

[桃園市]
龍潭區, 平鎮區, 大溪區, 八德區, 中壢區, 復興區, 新屋區, 楊梅區, 龜山區, 桃園區, 蘆竹區, 大園區, 觀音區

[新竹市]
香山區, 北區, 東區

[新竹縣]
五峰鄉, 關西鎮, 寶山鄉, 峨眉鄉, 竹北市, 北埔鄉, 芎林鄉, 竹東鎮, 橫山鄉, 尖石鄉, 新埔鎮, 湖口鄉, 新豐鄉

[苗栗縣]
通霄鎮, 頭屋鄉, 公館鄉, 苗栗市, 後龍鎮, 西湖鄉, 造橋鄉, 三灣鄉, 頭份市, 竹南鎮, 南庄鄉, 三義鄉, 大湖鄉, 泰安鄉, 卓蘭鎮, 獅潭鄉, 銅鑼鄉, 苑裡鎮

[臺中市]
龍井區, 東勢區, 大雅區, 大肚區, 北屯區, 北區, 太平區, 烏日區, 東區, 西屯區, 西區, 清水區, 中區, 霧峰區, 潭子區, 豐原區, 神岡區, 南屯區, 沙鹿區, 大里區, 南區, 和平區, 新社區, 石岡區, 后里區, 大甲區, 外埔區, 大安區, 梧棲區

[彰化縣]
線西鄉, 伸港鄉, 鹿港鎮, 和美鎮, 溪州鄉, 田中鎮, 二水鄉, 二林鎮, 福興鄉, 埔鹽鄉, 芳苑鄉, 秀水鄉, 芬園鄉, 員林市, 埤頭鄉, 北斗鎮, 田尾鄉, 社頭鄉, 彰化市, 大村鄉, 花壇鄉, 大城鄉, 竹塘鄉, 溪湖鎮, 永靖鄉, 埔心鄉

[南投縣]
竹山鎮, 名間鄉, 水里鄉, 信義鄉, 集集鎮, 鹿谷鄉, 草屯鎮, 南投市, 國姓鄉, 中寮鄉, 仁愛鄉, 埔里鎮, 魚池鄉

[雲林縣]
古坑鄉, 斗南鎮, 土庫鎮, 大埤鄉, 虎尾鎮, 元長鄉, 林內鄉, 莿桐鄉, 西螺鎮, 斗六市, 北港鎮, 水林鄉, 口湖鄉, 四湖鄉, 褒忠鄉, 崙背鄉, 二崙鄉, 麥寮鄉, 東勢鄉, 臺西鄉

[嘉義市]
東區, 西區

[嘉義縣]
新港鄉, 溪口鄉, 梅山鄉, 竹崎鄉, 大林鎮, 太保市, 民雄鄉, 阿里山鄉, 布袋鎮, 六腳鄉, 義竹鄉, 水上鄉, 中埔鄉, 大埔鄉, 番路鄉, 東石鄉, 朴子市, 鹿草鄉

[臺南市]
歸仁區, 安定區, 仁德區, 東區, 永康區, 安南區, 北區, 新市區, 新化區, 中西區, 學甲區, 左鎮區, 龍崎區, 南區, 鹽水區, 下營區, 東山區, 後壁區, 新營區, 柳營區, 六甲區, 白河區, 南化區, 西港區, 善化區, 大內區, 山上區, 麻豆區, 安平區, 佳里區, 七股區, 玉井區, 官田區, 楠西區, 關廟區, 將軍區, 北門區

[高雄市]
新興區, 鼓山區, 旗津區, 前金區, 苓雅區, 三民區, 前鎮區, 鹽埕區, 田寮區, 燕巢區, 內門區, 杉林區, 小港區, 鳳山區, 鳥松區, 大樹區, 旗山區, 美濃區, 六龜區, 茄萣區, 湖內區, 橋頭區, 梓官區, 大社區, 桃源區, 甲仙區, 茂林區, 仁武區, 岡山區, 楠梓區, 那瑪夏區, 阿蓮區, 路竹區, 永安區, 大寮區, 林園區, 彌陀區, 左營區

[屏東縣]
崁頂鄉, 林邊鄉, 東港鎮, 南州鄉, 潮州鎮, 佳冬鄉, 枋山鄉, 獅子鄉, 泰武鄉, 萬巒鄉, 新埤鄉, 來義鄉, 春日鄉, 枋寮鄉, 鹽埔鄉, 高樹鄉, 里港鄉, 九如鄉, 三地門鄉, 霧臺鄉, 恆春鎮, 牡丹鄉, 車城鄉, 滿州鄉, 琉球鄉, 屏東市, 內埔鄉, 長治鄉, 瑪家鄉, 麟洛鄉, 竹田鄉, 新園鄉, 萬丹鄉

[宜蘭縣]
大同鄉, 壯圍鄉, 礁溪鄉, 宜蘭市, 員山鄉, 五結鄉, 冬山鄉, 三星鄉, 羅東鎮, 頭城鎮, 南澳鄉, 蘇澳鎮

[花蓮縣]
玉里鎮, 豐濱鄉, 卓溪鄉, 富里鄉, 瑞穗鄉, 花蓮市, 秀林鄉, 新城鄉, 吉安鄉, 壽豐鄉, 萬榮鄉, 鳳林鎮, 光復鄉

[臺東縣]
長濱鄉, 卑南鄉, 延平鄉, 成功鎮, 鹿野鄉, 池上鄉, 東河鄉, 關山鎮, 海端鄉, 金峰鄉, 達仁鄉, 臺東市, 太麻里鄉, 大武鄉, 綠島鄉, 蘭嶼鄉

[澎湖縣]
湖西鄉, 馬公市, 白沙鄉, 西嶼鄉, 望安鄉, 七美鄉

【輸出格式】：
- 若同時符合上述兩項，輸出：
  - "Towns": 使用者輸入中有明確出現則擷取，否則為 "null"
- 若任一條件不符，輸出：
  - Counties 與 Towns 均設為 "null"
- 僅包含一個 JSON 物件，不能多餘文字。

【範例】：
使用者輸入：「萬巒」(意圖明顯提及「萬巒鄉」，屬於「屏東縣」的三級行政區。自動補全其所屬的一、二級行政區與單位。)
output:{"Counties": "屏東縣", "Towns": "萬巒鄉"}

使用者輸入：「台中」(意圖明顯提及「台中市」，屬於一、二級行政區。參考資料庫臺中市正楷名稱。)
output:{"Counties": "臺中市", "Towns": "null"}

使用者輸入：「屏東縣高雄」(意圖明顯提及「屏東縣」，但「高雄」並非其所屬的三級行政區，因此無效。)
output:{"Counties": "屏東縣", "Towns": "null"}

使用者輸入：「里萬區有停水嗎」(意圖明顯提及「里萬區」，但「里萬」並非有效的地名，無法對應到任何一、二級或三級行政區。)
output:{"Counties": "null", "Towns": "null"}

使用者輸入：「我想查停水」(意圖明顯提及停水，但沒有明確的地名。)
output:{"Counties": "null", "Towns": "null"}"""

        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# 普通對話機器人
class NormalLLM(ClassifierLLM):  # 可繼承同樣底層
    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": '根據現有資訊，使用中文回答使用者提出的問題。'},
                {"role": "user",   "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


#==========================
question_classifier = LLMChain(
    llm=ClassifierLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""
請判斷下面這段使用者輸入，是否為「提出問題」？
- 如果是在提問，回覆「是」
- 如果不是提問（閒聊、陳述等），回覆「否」
且回覆中僅包含一個字，不能多餘文字。

使用者：{text}"""
    )
)

can_answer_chain = LLMChain(
    llm=ClassifierLLM(),
    prompt=PromptTemplate(
        input_variables=["question", "docs"],
        template="""
下面是從本地知識庫檢索到的文件片段：
{docs}

請根據上述片段，判斷能否回答以下用戶提問：
「{question}」
- 如果能，僅回「是」
- 如果不能，僅回「否」
且回覆中僅包含一個字，不要其它多餘文字。"""
    ),
    output_key="verdict"
)

wrong_question_classifier = LLMChain(
    llm=ClassifierLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""
請判斷下面這段使用者輸入的問題，是否與水務業務相關？
- 如果是，回覆「是」
- 如果不是（閒聊、科普、陳述、非水務業務相關等），回覆「否」
且回覆中僅包含一個字，不能多餘文字。

使用者：{text}
"""
    )
)
#==========================

#llm_retrieve_chain = LLMChain(
#    llm=RetrieveLLM(),
#    prompt=PromptTemplate(
#        input_variables=["question", "docs"],
#        template="""下面是從本地知識庫檢索到的文件片段：
#{docs}
#
#請根據上述片段，請選擇一個最能解答使用者的疑問之文件片段：
#「{question}」
#
#且回覆中僅包含其中一個文件片段的內容，不要將編號與標題一同輸出，以及其它多餘文字。"""
#    ),
#    output_key="verdict"
#)
llm_retrieve_chain = LLMChain(
    llm=RetrieveLLM(),
    prompt=PromptTemplate(
        input_variables=["question", "docs"],
        template="""下面是從本地知識庫檢索到的文件片段：
{docs}

請根據上述片段，請選擇一個最能解答使用者的疑問之文件片段：
「{question}」

僅回覆解答文件片段所屬的整數編號，不要將標題、內容或符號一同輸出，以及其它多餘文字。"""
    ),
    output_key="verdict"
)

emotion_classifier = LLMChain(
    llm=EmotionLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""使用者：{text}"""
    )
)


location_outage_classifier = LLMChain(
    llm=LocationOutageLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""使用者：{text}"""
    )
)


status_classifier = LLMChain(
    llm=StatusLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""對話紀錄：{text}
        
使用者最新訊息：{user_message}"""
    )
)

normal_classifier = LLMChain(
    llm=NormalLLM(),
    prompt=PromptTemplate(
        input_variables=["text", "info"],
        template="""
使用者：{text}。
資訊：{info}。
現在時間：{time}。
"""
    )
)


CATEGORY_MAP = {
    1: "電子帳單、簡訊帳單及通知服務",
    2: "帳單與繳費管理",
    3: "用戶帳戶與用水設備管理",
    4: "水質、淨水與生活應用",
    5: "污水下水道與污水使用費",
    6: "緊急停水、計畫停水與應變",
    7: "水價政策與事業經營",
    8: "App／網站使用與隱私政策",
}


def generate_water_off_notification(no=None, start_date=None, end_date=None, start_time=None, end_time=None, 
                                  water_off_region=None, water_off_reason=None, water_off_number=None, 
                                  contact=None, pressure_down_region=None, pressure_down_reason=None, 
                                  pressure_down_number=None):
    """
    生成停水資訊通知的markdown模板
    
    參數:
    - no: 編號 (可選)
    - start_date: 開始日期 (格式: YYYY-MM-DD 或中文) (可選)
    - end_date: 結束日期 (格式: YYYY-MM-DD 或中文) (可選)
    - start_time: 開始時間 (格式: HH:MM 或中文) (可選)
    - end_time: 結束時間 (格式: HH:MM 或中文) (可選)
    - water_off_region: 停水區域 (可選)
    - water_off_reason: 停水原因 (可選)
    - water_off_number: 停水戶數 (可選)
    - contact: 聯絡電話 (可選)
    - pressure_down_region: 減壓區域 (可選)
    - pressure_down_reason: 減壓原因 (可選)
    - pressure_down_number: 減壓戶數 (可選)
    """
    
    # 格式化日期時間
    formatted_start_date = ''
    formatted_end_date = ''
    formatted_start_time = ''
    formatted_end_time = ''
    
    if start_date:
        if len(start_date) == 10 and start_date.count('-') == 2:
            formatted_start_date = start_date.replace('-', '年', 1).replace('-', '月') + '日'
        else:
            formatted_start_date = start_date
            
    if end_date:
        if len(end_date) == 10 and end_date.count('-') == 2:
            formatted_end_date = end_date.replace('-', '年', 1).replace('-', '月') + '日'
        else:
            formatted_end_date = end_date
    
    if start_time and ':' in start_time:
        hour, minute = start_time.split(':')
        formatted_start_time = f"上午{hour}:{minute}" if int(hour) < 12 else f"下午{int(hour)-12 if int(hour) > 12 else hour}:{minute}"
    elif start_time:
        formatted_start_time = start_time
    
    if end_time and ':' in end_time:
        hour, minute = end_time.split(':')
        formatted_end_time = f"上午{hour}:{minute}" if int(hour) < 12 else f"下午{int(hour)-12 if int(hour) > 12 else hour}:{minute}"
    elif end_time:
        formatted_end_time = end_time
    
    template = f"""## 停水通知"""

    # 添加編號（如果有）
    if no:
        template += f"（編號：[{no}](https://web.water.gov.tw/wateroffmap/map/view/{no})）"
        
    # 添加時間資訊（如果有）
    if formatted_start_date or formatted_end_date or formatted_start_time or formatted_end_time:
        template += f"""

### 📅 停水時間"""
        
        if formatted_start_date or formatted_end_date:
            template += f"\n- **日期**："
            if formatted_start_date and formatted_end_date:
                template += f"{formatted_start_date} 至 {formatted_end_date}"
            elif formatted_start_date:
                template += f"{formatted_start_date} 起"
            else:
                template += f"至 {formatted_end_date}"
                
        if formatted_start_time or formatted_end_time:
            template += f"\n- **時間**："
            if formatted_start_time and formatted_end_time:
                template += f"{formatted_start_time} 至 {formatted_end_time}"
            elif formatted_start_time:
                template += f"{formatted_start_time} 起"
            else:
                template += f"至 {formatted_end_time}"

    # 添加影響區域（如果有）
    if water_off_region:
        template += f"""

### 📍 影響區域
{water_off_region.replace("~", "至")}"""

    # 添加停水原因（如果有）
    if water_off_reason:
        template += f"""

### 🔧 停水原因
{water_off_reason}"""

    # 添加影響戶數（如果有）
    if water_off_number is not None:
        template += f"""

### 📊 影響戶數
**{water_off_number:,}戶**"""

    # 如果有減壓資訊，加入減壓部分
    if pressure_down_region or pressure_down_reason or (pressure_down_number is not None and pressure_down_number > 0):
        template += f"""

### ⚡ 減壓影響"""
        
        if pressure_down_region:
            template += f"\n- **減壓區域**：{pressure_down_region}"
            
        if pressure_down_reason:
            template += f"\n- **減壓原因**：{pressure_down_reason}"
            
        if pressure_down_number is not None:
            template += f"\n- **減壓戶數**：**{pressure_down_number:,}戶**"

    # 添加聯絡電話（如果有）
    if contact:
        template += f"""

### ☎️ 聯絡電話
**{contact}**"""

    # 添加注意事項
    template += f"""

---

"""

    return template

# 定義模板頭部
template_title = """# 🚰 [供水查詢](https://web.water.gov.tw/wateroffmap/map)

"""

# 定義模板尾部
template_note = """## ⚠️ 重要注意事項

1. **儲水準備**：停水範圍內用戶請自行儲水備用
2. **安全提醒**：停水期間請慎防火源，關閉抽水機電源
3. **防污染措施**：建築物自來水進水口低於地面的用戶，請關閉總表前制水閥
4. **復水時間**：管線末端及高地區域可能延遲復水
5. **進度查詢**：可至[停水查詢系統](https://web.water.gov.tw/wateroffmap/map)查詢停復水進度"""


class WaterGPTClient:
    def __init__(self):
        self.shared = {"last_docs": []}
        self.headers = HEADERS
        self.embedding_url = EMBEDDING_URL
        # 使用者是否詢問停水相關旗標
        self.water_outage_flag = False

        # 機器人狀態
        #  READY：準備就緒
        #  OUTAGE：停水查詢
        #  RAG：RAG查詢
        self.STATUS = "READY"  # 機器人狀態
        #self.OUTAGE_COUNTY = ""  # 停水查詢縣市
        #self.OUTAGE_TOWNS = ""  # 停水查詢鄉鎮市區

    # 移除WebSocket連接方法，改為直接使用requests
    async def ask(self, text, history, quick_replies=[]):
        text = text.strip()
        user_history = history.copy()  # 複製歷史對話，避免修改原始資料

        user_history.append({"role": "user", "content": text})

        history_str = []
        for entry in history:
            role = entry['role']
            content = entry['content']
            if role == 'system':
                continue  # 跳過 system 的內容
            history_str.append(f"{role}:{content}")

        # 用換行符連接結果
        formatted_string = '\n'.join(history_str)
        print(formatted_string)

        #print(history)
        status = status_classifier.predict(text=formatted_string, user_message=text).strip()
        status = status.replace("json", "").replace("```", "").replace("\n", "").replace(" ", "")

        # 情緒判斷
        emotion = emotion_classifier.predict(text=text).strip()

        if emotion == "anger":
            result = "非常抱歉讓您感到不滿意，我會盡快為您服務。"
            return result, history # 返回情緒回應, 不新增歷史對話
        
        print(status)
        status = json.loads(status)

        if status['status'] == "RAG":
            # 直接使用requests發送POST請求
            payload = {
                "request": text,
                "top_k": 5
            }
            response = requests.post(self.embedding_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            docs = data["response"]

            #if not docs:
            #    return "❌ 沒有找到相關文件。", history

            # 更新shared字典，保持與原代碼相容
            self.shared["last_docs"] = docs

            docs_text = "\n\n".join(
                f"{i+1} 標題：{d['title']}\n內容：{d['content']}"
                for i, d in enumerate(docs)
            )

            # 將每一個文件標題加入快捷訊息
            for d in docs:
                quick_replies.append(d['title'])

            #print(docs)
            # 判斷是否能回答
            answerable = can_answer_chain.predict(
                question=text,
                docs=docs_text
            ).strip()

            if answerable == "是":
                result = llm_retrieve_chain.predict(
                    question=text,
                    docs=docs_text
                ).strip()

                #try:
                result = docs[int(result)-1]['content'] 
                #except:
                #    return "❌ 無法獲取正確的文件編號，請稍後再試。", history
            
                user_history.append({"role": "assistant", "content": "(RAG內容)"})

                return result, user_history
            else:
                # 判斷是否為水務相關問題
                wrong_question = wrong_question_classifier.predict(text=text).strip()
                if wrong_question == "是":
                    return "✔ 我可以幫你接洽專人", history # 不新增歷史對話
                else:
                    return "✘ 很抱歉，請詢問與台灣自來水公司相關之問題喔!", history # 不新增歷史對話
                
        if status['status'] == "OUTAGE":
            location_outage_str = location_outage_classifier.predict(text=text).strip()
            location_outage_str = location_outage_str.replace("json", "").replace("```", "").replace("\n", "").replace(" ", "")

            try:
                print(location_outage_str)
                location = json.loads(location_outage_str)
                water_affected_counties = location['Counties']
                water_affected_towns = location['Towns']

                if water_affected_towns == "null":
                    water_affected_towns = None

                if water_affected_counties == "null":
                    user_history.append({"role": "assistant", "content": "請輸入詳細地區，例如：台中市北區"})
                    return "請輸入詳細地區，例如：台中市北區", user_history

                response = requests.get(WATER_OUTAGE_URL, params={"affectedCounties": water_affected_counties, "affectedTowns": water_affected_towns, "query": "name"})
                
                response = response.json()

                if response.get("message") == "success":
                    response = response.get("result")
                else:
                    return "伺服器忙碌中，請稍後再試。", history # 不新增歷史對話

                output = ""
                for i in response:
                    output += generate_water_off_notification(
                        no=i["no"],
                        start_date=i["startDate"],
                        end_date=i["endDate"],
                        start_time=i["startTime"],
                        end_time=i["endTime"],
                        water_off_region=i["waterOffRegion"],
                        water_off_reason=i["waterOffReason"],
                        water_off_number=i["waterOffNumber"],
                        contact=i["contact"],
                        pressure_down_region=i["pressureDownRegion"],
                        pressure_down_reason=i["pressureDownReason"],
                        pressure_down_number=i["pressureDownNumber"],
                    )
                user_history.append({"role": "assistant", "content": "(回應停水內容)"})
                if output == "":
                    if water_affected_towns != None:
                        return f"✅ 目前{water_affected_counties}{water_affected_towns}地區無停水資訊，如有用水問題請撥本公司24小時免付費客服專線『1910』。", user_history
                    else:
                        return f"✅ 目前{water_affected_counties}地區無停水資訊，如有用水問題請撥本公司24小時免付費客服專線『1910』。", user_history
                
                return template_title + output + template_note, user_history

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic string that caused error: ---{e.doc}---") # e.doc 是導致錯誤的原始字串
                return "您輸入的資訊有誤，請稍後再試。", history # 不新增歷史對話

        return "✘ 這看起來不是一個問題，請輸入水務相關提問。", history # 不新增歷史對話

# 移除原來的handle_ws函數，改為直接請求的函數
async def get_embedding_data(text, top_k=5):
    payload = {
        "request": text,
        "top_k": top_k
    }
    response = requests.post(EMBEDDING_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["response"]


async def main():
    print("Bot ready，輸入 exit 離開。")

    shared = {"last_docs": []}

    while True:
        text = await asyncio.to_thread(input, "> ")
        text = text.strip()
        if text.lower() in ("exit", "quit"):
            break

        verdict = question_classifier.predict(text=text).strip()
        print(f"[分類結果] {verdict}")

        if verdict != "是":
            print("✘ 這看起來不是一個問題，請隨時輸入水務相關提問。")
            continue

        # 使用requests直接獲取資料
        try:
            docs = await get_embedding_data(text)
            shared["last_docs"] = docs

            print(f"⟳ 找到 {len(docs)} 篇最相關文件：")
            for i in docs:
                print(f"[score{i['confidence']}｜類別{i['category']}｜{CATEGORY_MAP[int(i['category'])]}] {i['title']}")
        except Exception as e:
            print(f"❌ 獲取嵌入數據時出錯: {e}")
            continue

        if not docs:
            print("❌ 沒有找到相關文件。")
            continue

        docs_text = "\n\n".join(
            f"[{i+1}] 標題：{d['title']}\n內容：{d['content']}"
            for i, d in enumerate(docs)
        )

        answerable = can_answer_chain.predict(
            question=text,
            docs=docs_text
        ).strip()

        if answerable == "是":
            result = llm_retrieve_chain.predict(
                question=text,
                docs=docs_text
            ).strip()
            print(result)
            continue

        wrong_question = wrong_question_classifier.predict(text=text).strip()
        if wrong_question == "是":
            print("✔ 我可以幫你接洽專人")
        else:
            print("✘ 很抱歉，請詢問與台灣自來水公司相關之問題喔!")

if __name__ == "__main__":
    pass

# WaterGPTClient 測試
'''
async def example():
    # 建立客戶端
    client = WaterGPTClient()
    
    # 提問
    response = await client.ask("請問如何繳水費？")
    print(f"回答: {response}")
    
    # 可以多次提問
    response = await client.ask("水質檢測標準是什麼？")
    print(f"回答: {response}")

# 運行範例
# asyncio.run(example())
'''