import os
import sys
import time
from fastapi import FastAPI, BackgroundTasks
from ovon.app.v1.server import router as v1_router

app = FastAPI()
app.include_router(v1_router, prefix="/api/v1")

def restart_process():
    """
    等待 1 秒后重启当前进程
    """
    time.sleep(1)  # 给客户端留出接收响应的时间
    print("\n--- 正在重启服务... ---")
    
    # 获取当前运行的解释器路径和所有命令行参数
    executable = sys.executable
    args = sys.argv
    
    # 用新进程替换当前进程
    os.execv(executable, [executable] + args)

@app.post("/restart")
async def restart(background_tasks: BackgroundTasks):
    # 将重启逻辑放入后台任务，这样 FastAPI 可以先返回 HTTP 响应
    background_tasks.add_task(restart_process)
    return {"status": "success", "message": "Server is restarting..."}

if __name__ == "__main__":
    import uvicorn
    # 注意：做重启 Demo 时，建议 reload 设为 False，避免 uvicorn 自身的监控干扰
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
