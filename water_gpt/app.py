from flask import *
# from flask_cors import CORS
from tools import ChatBot
from rag_api import send_to_websocket_sync

app = Flask(__name__, static_folder='static',  # 靜態檔案資料夾
            static_url_path='/static',)  # 靜態檔案對應網址
# CORS(app)  # 允許跨域請求

# 初始化聊天機器人
chatbot = ChatBot()

# 儲存訊息的列表
messages = []

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/send", methods=["POST"])
def send():
    data = request.json  # 解析message: userInput.value,並轉換成dit
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # 儲存用戶訊息
    messages.append({"role": "user", "message": user_message})

    # 根據使用者的問題查詢向量資料庫，返回第0個 data
    rag_result = send_to_websocket_sync(user_message)
    rag_result = rag_result[0]
    rag_content = "title:'''{}'''\n\n\ncontent:'''{}'''".format(rag_result["title"], rag_result["content"])
    
    # 使用 ChatBot 生成回答
    bot_reply = chatbot.chat_with_llm(f"user_message:'''{user_message}'''\n\n\nrag_content:'''{rag_content}'''")
    
    # 儲存 AI 回覆
    messages.append({"role": "bot", "message": bot_reply})

    return jsonify({"reply": f"生成回答: {bot_reply}"})


@app.route("/messages", methods=["GET"])
def get_messages():
    """ 取得所有聊天記錄 """
    return jsonify(messages)


@app.route("/clear", methods=['POST'])
def clear():
    global messages
    count = len(messages)
    messages = []
    # 重置聊天機器人歷史
    chatbot.history = [{
        "role": "system",
        "content": "你是一名客服幫助使用者解決問題"
    }]
    return (f" 刪除 {count} 筆資料")


@app.route("/quick_messages", methods=["GET"])
def quick_messages():
    """ 取得快捷訊息 """
    quick_replies = ["測試", "這是快捷訊息1", "這是快捷訊息2", "這是快捷訊息3"]
    return jsonify(quick_replies)


if __name__ == "__main__":
    app.run(debug=True)