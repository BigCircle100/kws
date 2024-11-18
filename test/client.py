# import asyncio
# import websockets

# async def send_data():
#     uri = "ws://127.0.0.1:8000/ws"
#     async with websockets.connect(uri) as websocket:
#         for i in range(1000):  # 发送0到10
#             await websocket.send(str(i))
#             print(f"Sent data: {i}")
#             await asyncio.sleep(0.2)  # 每200毫秒发送一次

# if __name__ == "__main__":
#     asyncio.run(send_data())


import httpx

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8000/api/data")
        print(response)

# 运行客户端
import asyncio

if __name__ == "__main__":
    asyncio.run(fetch_data())
