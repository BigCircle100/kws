# from fastapi import FastAPI, WebSocket, Response
# import asyncio
# import time

# app = FastAPI()


# @app.get("/api/empty")
# async def empty_response():
#     return Response(status_code=200)  # 仅返回状态码200，内容为空


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_text()
#             time.sleep(1)
#             print(f"Received data: {data}")
#     except Exception as e:
#         print(f"Connection closed: {e}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/api/data")
async def get_data():
    print(Response(status_code=200))
    return Response(status_code=200)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
