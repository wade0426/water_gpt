import requests
import json

# LLM_URL = "http://3090p8080.huannago.com/v1/chat/completions"
LLM_URL = "http://4090p8080.huannago.com/v1/chat/completions"

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

    def chat_with_llm(self, user_message):
        """與LLM進行對話"""
        self.history.append({"role": "user", "content": user_message})

        data = {
            "model": self.model,
            "messages": self.history,
            "stream": True
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=data, stream=True)
            response.encoding = 'utf-8'  # 正確解碼

            print("Assistant:", end=" ", flush=True)
            full_reply = ""

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                line = line[len("data: "):].strip()
                if line == "[DONE]":
                    break
                try:
                    payload = json.loads(line)
                    delta = payload["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                    full_reply += delta
                except json.JSONDecodeError as e:
                    print(f"\n[Error parsing line]: {line} => {e}")

            print()  # newline after assistant response
            self.history.append({"role": "assistant", "content": full_reply})
            return full_reply
        except Exception as e:
            error_msg = f"[Request Error]: {e}"
            print(error_msg)
            return error_msg

    def run_interactive_session(self):
        """運行互動式會話"""
        print("開始與AI聊天，輸入'exit'或'quit'結束對話")
        while True:
            user_message = input("> ")
            if user_message.lower() in ['exit', 'quit']:
                print("結束對話")
                break
            self.chat_with_llm(user_message)


# 示範如何使用此類別
if __name__ == "__main__":
    bot = ChatBot()
    bot.run_interactive_session()