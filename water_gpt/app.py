# pip install "Flask[async]"
from flask import *
# from flask_cors import CORS
from tools import ChatBot


app = Flask(__name__, static_folder='static',  # 靜態檔案資料夾
            static_url_path='/static',)  # 靜態檔案對應網址
# CORS(app)  # 允許跨域請求

# 初始化聊天機器人
chatbot = ChatBot()

# 儲存訊息的列表
messages = []

# 儲存快捷訊息的列表
quick_replies = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/send", methods=["POST"])
async def send():
    data = request.json  # 解析message: userInput.value,並轉換成dit
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # 儲存用戶訊息
    messages.append({"role": "user", "message": user_message})
    
    # 使用 ChatBot 生成回答
    bot_reply = await chatbot.chat_with_llm(user_message, quick_replies)
    
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
async def quick_messages():
    """ 取得快捷訊息 """
    if len(messages) < 2:
        return jsonify(["如何繳水費?", "什麼是簡訊帳單?", "如何查詢水質?"])
    else:
        # quick_replies = await chatbot.generate_quick_messages(messages)
        return jsonify(quick_replies)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    pass