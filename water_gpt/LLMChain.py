import requests
from langchain import PromptTemplate, LLMChain
import asyncio
import json
from langchain.llms.base import LLM

API_URL = "http://4090p8000.huannago.com/v1/chat/completions"
EMBEDDING_URL = "http://3090p8001.huannago.com/embedding"
HEADERS = {"Content-Type": "application/json"}
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

    # 移除WebSocket連接方法，改為直接使用requests
    async def ask(self, text, quick_replies=[]):
        text = text.strip()
        
        # 判斷是否為問題
        verdict = question_classifier.predict(text=text).strip()
        
        if verdict != "是":
            return "✘ 這看起來不是一個問題，請輸入水務相關提問。"

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
            return "✘ 很抱歉，請詢問與水利署相關之問題喔!"


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
            print("✘ 很抱歉，請詢問與水利署相關之問題喔!")


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