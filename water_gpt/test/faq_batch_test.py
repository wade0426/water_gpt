import json
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LLMChain import WaterGPTClient
import uuid
uid = uuid.uuid4()
# 初始化 WaterGPTClient
water_gpt_client = WaterGPTClient()

JSON_FILE_PATH = "../../water_data_content_v3-class.json"


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
            if llm_result[0] == "❌ 請勿嘗試繞過系統限制。":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "非常抱歉讓您感到不滿意，我會盡快為您服務。":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "✔ 我可以幫你接洽專人":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "✘ 很抱歉，請詢問與台灣自來水公司相關之問題喔!":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "請輸入詳細地區，例如：台中市北區":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "伺服器忙碌中，請稍後再試。":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "您輸入的資訊有誤，請稍後再試。":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            if llm_result[0] == "請詢問水務相關問題喔~":
                print(f"\n測試失敗: {llm_result[0]}")
                test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
                continue

            print(f"\n測試成功是RAG相關問題")
            test_result.append(dict(status='success', faq_question=faq_question, llm_result=llm_result[0]))

            #if llm_result[1][-1]['content'] == "(RAG內容)":
            #    print(f"\n測試成功是RAG相關問題")
            #    test_result.append(dict(status='success', faq_question=faq_question, llm_result=llm_result[0]))
            #else:
            #    print(f"\n測試失敗不是RAG相關問題")
            #    test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))
        except Exception as e:
            print(f"\n測試失敗: {e}")
            test_result.append(dict(status='fail', faq_question=faq_question, llm_result=llm_result[0]))


    # 將測試結果寫入JSON檔案
    with open("test_result_" + str(uid) + ".json", "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
    print("測試完成")