import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLMChain import WaterGPTClient

# 初始化 WaterGPTClient
water_gpt_client = WaterGPTClient()

JSON_FILE_PATH = "./water_gpt/water_data_content_v3.json"


async def main():
    # 讀取JSON檔案
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 測試的結果
    test_result = []

    for item in data:
        faq_question = item["title"]
        # print(faq_question)
        llm_result = await water_gpt_client.ask(faq_question, [])

        try:
            if llm_result[1][-1]['content'] == "(RAG內容)":
                print(f"\n測試成功是RAG相關問題")
                test_result.append(dict(status='success', faq_question=faq_question, llm_result=llm_result[0]))
            else:
                print(f"\n測試失敗不是RAG相關問題")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
        except Exception as e:
            print(f"\n測試失敗: {e}")
            test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))


    # 將測試結果寫入JSON檔案
    with open("test_result.json", "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
    print("測試完成")