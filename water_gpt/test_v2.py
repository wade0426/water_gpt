from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# llama.cpp 伺服器地址和Port
# LLAMA_CPP_SERVER_URL = "http://localhost:8080/v1"
LLAMA_CPP_SERVER_URL = "http://localhost:8080/v1/chat/completions"

# llama.cpp server 啟動時設定了 --api-key
# API_KEY = "YOUR_OPTIONAL_API_KEY"
# 如果沒有設定 --api-key，則 API_KEY 可以是任意非空字串，或者省略
API_KEY = "YOUR_OPTIONAL_API_KEY"

llm = ChatOpenAI(
    model_name="Qwen3-8B-Q4_K_M",
    openai_api_key=API_KEY,
    openai_api_base=LLAMA_CPP_SERVER_URL,
    temperature=0.7
)

try:
    response = llm.invoke([
        HumanMessage(content="你是誰?")
    ])
    print(response.content)

    # response_en = llm.invoke([
    #     HumanMessage(content="Who are you?")
    # ])
    # print(response_en.content)

except Exception as e:
    print(f"發生錯誤: {e}")
    print("請確認：")
    print(f"1. llama.cpp 伺服器是否已在 {LLAMA_CPP_SERVER_URL.replace('/v1','')} 啟動並正在運行。")
    print(f"2. 如果 llama.cpp 伺服器設定了 API Key，這裡的 API_KEY ('{API_KEY}') 是否正確。")
    print(f"3. model_name 是否與 llama.cpp 伺服器期待的相符（通常是載入的模型文件名）。")