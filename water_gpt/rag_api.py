import asyncio
import websockets
import json

async def connect_websocket():
    """建立WebSocket連接"""
    while True:
        try:
            return await websockets.connect("wss://3090p8001.huannago.com/ws/embedding")
        except ConnectionRefusedError:
            print("無法連接到服務器 - 重試中...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"連接時發生錯誤: {e} - 重試中...")
            await asyncio.sleep(5)

async def send_to_websocket(request):
    try:
        websocket = await connect_websocket()
        await websocket.send(json.dumps({"request": request, "top_k": 5}))
        response = await websocket.recv()
        docs = json.loads(response)
        docs = docs["response"]

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
        print(f"⟳ 找到 {len(docs)} 篇最相關文件：")

        for i in docs:
            print(f"[類別{i['category']}｜{CATEGORY_MAP[int(i['category'])]}] {i['title']}")
        
        # print(docs)
        return docs

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket連接已關閉")
    except Exception as e:
        print(f"發生錯誤: {e}")
        return f"發生錯誤: {e}"
    

# 同步版本的函數給 Flask 用
def send_to_websocket_sync(request):
    """同步版本的 send_to_websocket 函數"""
    return asyncio.run(send_to_websocket(request))


if __name__ == "__main__":
    asyncio.run(send_to_websocket(input(">")))
