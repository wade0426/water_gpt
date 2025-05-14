import requests
from langchain import PromptTemplate, LLMChain
import asyncio
import websockets
from langchain.llms.base import LLM
import json

API_URL = "http://4090p8000.huannago.com/v1/chat/completions"
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
                {"role": "system", "content": "你是一個訊息分類器，只回覆單字 “是” 或 “否”"},
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

async def handle_ws(ws, shared):
    while True:
        try:
            # 核心的接收訊息操作
            response = await ws.recv()

            # 如果收到 ping，回傳 pong (心跳機制)
            if response == "__ping__":
                await ws.send("__pong__")
                continue

            data = json.loads(response)
            docs = data["response"]
            # 共享數據操作 讓 main 可以拿到
            shared["last_docs"] = docs

            print(f"⟳ 找到 {len(docs)} 篇最相關文件：")
            for i in docs:
                print(f"[score{i['confidence']}｜類別{i['category']}｜{CATEGORY_MAP[int(i['category'])]}] {i['title']}")

        except websockets.ConnectionClosed:
            print("❌ WebSocket 已斷線")
            break

async def main():
    print("Connecting to WebSocket…")
    async with websockets.connect(
        "wss://3090p8001.huannago.com/ws/embedding",
        ping_interval=None  # 我們自己做 ping/pong
    ) as ws:
        print("WS connected, ready.")
        print("Bot ready，輸入 exit 離開。")

        shared = {"last_docs": []}
        # ws = WebSocket 連線物件
        ws_task = asyncio.create_task(handle_ws(ws, shared))

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

            await ws.send(json.dumps({"request": text, "top_k": 5}))
            await asyncio.sleep(1)  # 等 ws handler 收到結果
            docs = shared["last_docs"]

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

        ws_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())

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