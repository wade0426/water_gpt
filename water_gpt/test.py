# from ollama import chat
# stream = chat(model='llama3.2', messages=[
#     {
#         'role': 'user',
#         'content': '請你扮演農民的角色，描述他的價值觀',
#     },
#     # options代表模型生成回答的隨機性，0-1 預設為0.8
# ], stream=False, options={"temperature": 0.8})
# # stream 是否分段生成,預設為一次生成
# print(stream)
# # for chunk in stream:
# #     print(chunk.message.content, end='',flush=True)#flush 參數 用來強制刷新輸出緩衝區

# ---

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model_name="gemini-2.5-flash-preview-04-17",
    openai_api_key="AI",
    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.7
)

response = llm.invoke([
    HumanMessage(content="你是誰?")
])

print(response.content)