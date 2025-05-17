import json
import asyncio
from LLMChain import WaterGPTClient

# 初始化 WaterGPTClient
water_gpt_client = WaterGPTClient()

# LLM_URL = "http://3090p8080.huannago.com/v1/chat/completions"
LLM_URL = "http://4090p8080.huannago.com/v1/chat/completions"

QUICK_MESSAGES_PROMPT = """我將提供一段「對話歷史紀錄」。請你根據這段歷史紀錄，特別是**機器人最後的回應**，設想**使用者**接下來可能會提出的 **4 個相關問題**。

你的任務是：
1.  仔細分析機器人提供的資訊。
2.  思考使用者可能會對哪些細節感到好奇、需要進一步澄清，或是可能引申出的其他相關疑問。
3.  生成的問題應該是自然且合乎邏輯的延伸。

請將這 4 個預測的問題以 JSON 格式輸出，結構如下：
{
  "questions": [
    "問題一",
    "問題二",
    "問題三",
    "問題四"
  ]
}

請嚴格遵守此 JSON 格式。
"""

class ChatBot:
    def __init__(self, url=LLM_URL, model="your-model-name"):
        self.url = url
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        self.history = [{
            "role": "system",
            "content": "你是一名客服幫助使用者解決問題"
        }]

    def set_system_prompt(self, system_prompt):
        """設定系統提示詞"""
        self.history = [{
            "role": "system",
            "content": system_prompt
        }]

    async def chat_with_llm(self, user_message, quick_replies=[]):
        """與LLM進行對話"""
        result = await water_gpt_client.ask(user_message, quick_replies)
        return result


    def run_interactive_session(self):
        """運行互動式會話"""
        print("開始與AI聊天，輸入'exit'或'quit'結束對話")
        while True:
            user_message = input("> ")
            if user_message.lower() in ['exit', 'quit']:
                print("結束對話")
                break
            self.chat_with_llm(user_message)


    async def generate_quick_messages(self, history):
        """生成快速訊息"""
        history = str(history)
        generate_quick_messages_result = await water_gpt_client.generate_quick_messages(history)
        # # 去除引號
        generate_quick_messages_result = generate_quick_messages_result.replace('`', '').replace('json', '').replace(' ', '').replace('\n', '')
        try:
            # 將 json 轉換成 dict
            generate_quick_messages_result = json.loads(generate_quick_messages_result)
            # 將 dict 中的 questions 取出
            questions = generate_quick_messages_result["questions"]
            return questions
        except Exception as e:
            print(f"生成快速訊息時發生錯誤: {e}")
            print(f"json: {generate_quick_messages_result}")
            return ["如何繳水費?", "什麼是簡訊帳單?", "如何查詢水質?"]


async def test():
    # 初始化快速訊息機器人
    quick_messages_bot = ChatBot()
    messages = "[{'role': 'user', 'message': '如何繳水費?'}, {'role': 'bot', 'message': '您好，繳水費的方式有很多種，您可以參考以下方式：\n\n1. **親自繳費：** 請您直接到我們當地的服務、營運所繳費。\n2. **超商繳費：** 在 統一、全家、OK、萊爾富超商的多媒體機（例如ibon）上，輸入水號查詢並列印繳費單，然後在超商櫃檯繳費。\n3. **行動支付：** 您可以使用街口支付、Pi拍錢包、LINE Pay等行動支付APP，直接輸入水號繳納水費。\n4. **線 上繳費：** 您可以到我們的網站首頁，點選「線上繳費」，選擇「信用卡繳費」或「網路繳費」，線上進行繳費。\n\n**注意事項：**\n\n*   水費單的代收期限是次月21日，如果在期限內還未收到通知單，但仍在代收期限內，您 可以申請補單。\n*   如果前期水費未繳，請儘早前往我們的服務所繳費，以免停水。\n*   二期催繳欠費不開放補寄帳單喔。\n\n希望這些資訊對您有幫助！'}]"
    result = await quick_messages_bot.generate_quick_messages(messages)
    print(f"result: {result}")
    water_gpt_client.disconnect()


# 示範如何使用此類別
if __name__ == "__main__":
    # bot = ChatBot()
    # bot.run_interactive_session()

    # asyncio.run(test())
    pass