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
        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": "你是一個情緒辨識器，只回覆情緒標籤"},
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
        payload = {
            "model":    MODEL,
            "messages": [
                {"role": "system", "content": '你只是一個停水查詢器，判斷使用者是否在詢問停水資訊，請回覆JSON格式 輸出範例 {"result": "true", "affectedCounties": "臺中市", "affectedTowns": "北區", "query": "name"}，不要輸出與JSON格式無關的文字'},
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


question_classifier = LLMChain(
    llm=ClassifierLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""
請判斷下面這段使用者輸入，是否為「提出問題」？
- 如果是在提問，回覆「是」
- 如果不是提問（閒聊、陳述等），回覆「否」
且回覆中僅包含一個字，不能多餘文字。

使用者：{text}
"""
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
且回覆中僅包含一個字，不要其它多餘文字。
"""
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


llm_retrieve_chain = LLMChain(
    llm=RetrieveLLM(),
    prompt=PromptTemplate(
        input_variables=["question", "docs"],
        template="""
下面是從本地知識庫檢索到的文件片段：
{docs}

請根據上述片段，請選擇一個最能解答使用者的疑問之文件片段：
「{question}」

且回覆中僅包含其中一個文件片段的內容，不要將編號與標題一同輸出，以及其它多餘文字。
"""
    ),
    output_key="verdict"
)


emotion_classifier = LLMChain(
    llm=EmotionLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""您是一個高度專業的情緒辨識雷達。情境是一位專業保險業務員的對話文本，您的任務是分析輸入的文本，請嚴格遵守以下指南:

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

請根據以上指南,準確地將輸入文本歸類為5種情緒之一。

使用者：{text}
"""
    )
)


water_outage_classifier = LLMChain(
    llm=WaterOutageLLM(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="""
使用者：{text}。

請根據上述訊息，輸出JSON格式
範例輸出：
{{"result": "true", "affectedCounties": "臺中市", "affectedTowns": "北區", "query": "name"}} 
{{"result": "false", "affectedCounties": "null", "affectedTowns": "null", "query": "null"}}。
"""
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

        water_outage_str = water_outage_classifier.predict(text=text).strip()
        water_outage_str = water_outage_str.replace("json", "").replace("```", "").replace("\n", "").replace(" ", "")
        try:
            data = json.loads(water_outage_str)
            print("JSON decoded successfully:", data)

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

        # 判斷是否為問題
        verdict = question_classifier.predict(text=text).strip()
        
        if verdict != "是":
            return "✘ 這看起來不是一個問題，請輸入水務相關提問。"

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