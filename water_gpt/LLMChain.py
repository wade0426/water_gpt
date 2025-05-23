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


class RetrieveLLM(ClassifierLLM):  # 可繼承同樣底層
    def _call(self, prompt: str, stop=None) -> str:
        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": "你是一個文件片段選擇器，只回覆文件片段內存在的內容"},
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


# 停水查詢
class WaterOutageLLM(ClassifierLLM):  # 可繼承同樣底層
    def _call(self, prompt: str, stop=None) -> str:
        system_prompt = """
請判斷使用者輸入的訊息是否具有「明確的停水資訊查詢需求」，並遵守以下規則：

【判斷條件】需同時符合：
1. 訊息中必須明確提及台灣地名（由使用者輸入直接提到），地名可分為：
   - affectedCounties：一、二級行政區（如「臺中市」、「南投縣」）
   - affectedTowns：三級行政區（如「北區」、「埔里鎮」）
   如果提及到多個一、二級行政區，則擷取首個
   如果提及到多個三級行政區，則擷取首個
   - 參考地名資料庫：
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

2. 訊息中必須包含停水查詢意圖，例如詞句包含：
   - 「停水」、「供水」、「幾點會來水」、「什麼時候會有水」、「水來了沒」、「停水公告」、「水什麼時候來」、「供水狀況」等

【輸出格式】：
- 若同時符合上述兩項，輸出：
  - "result": "true"
  - "affectedTowns": 使用者輸入中有明確出現則擷取，否則為 "null"
- 若任一條件不符，輸出：
  - "result": "false"
  - affectedCounties 與 affectedTowns 均設為 "null"
- 所有結果都需加上：
  - "query": "name"
- 僅包含一個 JSON 物件，不能多餘文字。

【範例】：
使用者輸入：「萬巒有沒有停水」  
✅ 有停水查詢意圖  
✅ 只有提到「萬巒」  
➡ 輸出：
{"result": "true", "affectedCounties": "屏東縣", "affectedTowns": "萬巒鄉", "query": "name"}

使用者輸入：「北區今天會停水嗎」  
➡ 輸出：
{"result": "true", "affectedCounties": "臺中市", "affectedTowns": "北區", "query": "name"}

使用者輸入：「屏東縣供水狀況？」  
➡ 輸出：
{"result": "true", "affectedCounties": "屏東縣", "affectedTowns": "null", "query": "name"}

使用者輸入：「台中供水狀況？」  
➡ 輸出：
{"result": "true", "affectedCounties": "臺中市", "affectedTowns": "null", "query": "name"}

使用者輸入：「萬巒」  
❌ 沒有停水查詢意圖  
➡ 輸出：
{"result": "false", "affectedCounties": "null", "affectedTowns": "null", "query": "name"}"""

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

llm_retrieve_chain = LLMChain(
    llm=RetrieveLLM(),
    prompt=PromptTemplate(
        input_variables=["question", "docs"],
        template="""下面是從本地知識庫檢索到的文件片段：
{docs}

請根據上述片段，請選擇一個最能解答使用者的疑問之文件片段：
「{question}」

且回覆中僅包含其中一個文件片段的內容，不要將編號與標題一同輸出，以及其它多餘文字。"""
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

water_outage_classifier = LLMChain(
    llm=WaterOutageLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""使用者：{text}"""
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


class WaterGPTClient:
    def __init__(self):
        self.shared = {"last_docs": []}
        self.headers = HEADERS
        self.embedding_url = EMBEDDING_URL
        # 使用者是否詢問停水相關旗標
        self.water_outage_flag = False

    # 移除WebSocket連接方法，改為直接使用requests
    async def ask(self, text, quick_replies=[]):
        text = text.strip()

        emotion = emotion_classifier.predict(text=text).strip()

        if emotion == "anger":
            return "非常抱歉讓您感到不滿意，我會盡快為您服務。"

        # 判斷是否為問題
        verdict = question_classifier.predict(text=text).strip()
        
        if verdict != "是":
            return "✘ 這看起來不是一個問題，請輸入水務相關提問。"

        # 判斷是否為水務相關問題
        water_outage_str = water_outage_classifier.predict(text=text).strip()
        print(water_outage_str)
        water_outage_str = water_outage_str.replace("json", "").replace("```", "").replace("\n", "").replace(" ", "")
        try:
            data = json.loads(water_outage_str)
            #print("JSON decoded successfully:", data)

            water_result = data.get("result")
            water_affected_counties = data.get("affectedCounties")
            water_affected_towns = data.get("affectedTowns")

            if water_affected_towns == "null":
                water_affected_towns = None
            
            # print(water_result)
            # print(water_affected_counties)
            # print(water_affected_towns)

            if water_result == "true":
                self.water_outage_flag = True
                response = requests.get(WATER_OUTAGE_URL, params={"affectedCounties": water_affected_counties, "affectedTowns": water_affected_towns, "query": "name"})
                result = normal_classifier.predict(
                    text=text,
                    info=response.json(),
                    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                return result

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic string that caused error: ---{e.doc}---") # e.doc 是導致錯誤的原始字串
            return "您輸入的資訊有誤，請稍後再試。"


        # 直接使用requests發送POST請求
        payload = {
            "request": text,
            "top_k": 5
        }
        response = requests.post(self.embedding_url, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        docs = data["response"]
        
        if not docs:
            return "❌ 沒有找到相關文件。"

        # 更新shared字典，保持與原代碼相容
        self.shared["last_docs"] = docs

        docs_text = "\n\n".join(
            f"[{i+1}] 標題：{d['title']}\n內容：{d['content']}"
            for i, d in enumerate(docs)
        )

        # 將每一個文件標題加入快捷訊息
        for d in docs:
            quick_replies.append(d['title'])

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
            return result



        # 判斷是否為水務相關問題
        wrong_question = wrong_question_classifier.predict(text=text).strip()
        if wrong_question == "是":
            return "✔ 我可以幫你接洽專人"
        else:
            return "✘ 很抱歉，請詢問與台灣自來水公司相關之問題喔!"


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
    # asyncio.run(main())

    # payload = {
    #     "model":    MODEL,
    #     "messages": [
    #         {"role": "system", "content": "你是一個客服，請回答使用者提出的問題。"},
    #         {"role": "user",   "content": "請問如何繳水費？"}
    #     ],
    #     "stream": False
    # }
    # resp = requests.post(API_URL, headers=HEADERS, json=payload)
    # resp.raise_for_status()
    # data = resp.json()
    # print(data["choices"][0]["message"]["content"])
    
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