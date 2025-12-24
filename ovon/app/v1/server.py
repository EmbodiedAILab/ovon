import os
import json
from datetime import datetime
from fastapi import APIRouter,BackgroundTasks
from pydantic import BaseModel

from ovon.app.v1.demo1230 import OVON2Limo, create_episode

router = APIRouter()


class NavigationRequest(BaseModel):
    task: str

template_episode_file = "/workspace/ovon/data/task_datasets/objectnav/cloudrobo_v1/demo_1230/val/content/shenzhen-room_ziwei_20250724-metacam.json.gz"
start_position =  [
                    -0.016438689082860947,
                    0.05,
                    4.700855445861816
                ]
start_rotation = [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]


# 定义后台执行的函数
def run_navigation_task(obs_dir: str, object_category: str):
    try:
        # 1. 初始化剧本
        create_episode(template_episode_file, start_position, start_rotation, object_category)

        # 2. 运行环境
        env = OVON2Limo(obs_dir, object_category)
        finish = False
        while not finish:
            finish = env.act()
        
        print(f"Task {obs_dir} Finished!")

        # 3. 写入结果文件
        # 确保目录存在
        os.makedirs(obs_dir, exist_ok=True)
        result_path = os.path.join(obs_dir, "result.txt")
        with open(result_path, "w") as f:
            json.dump({"status": "SUCCESS"}, f)
            
    except Exception as e:
        print(f"Task failed: {e}")
        # 如果失败，也可以记录失败状态
        result_path = os.path.join(obs_dir, "result.txt")
        with open(result_path, "w") as f:
            json.dump({"status": "FAILED", "error": str(e)}, f)

def gen_save_dir_obs_dir(task, timestamp):
    save_dir = f"/workspace/ovon/ovon/app/v1/output/{task}/{timestamp}"
    obs_dir = f"output/{task}/{timestamp}"
    return save_dir, obs_dir


@router.post("/navigation")
def get_navigation(request: NavigationRequest, background_tasks: BackgroundTasks):
    # 生成动态路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir, obs_dir = gen_save_dir_obs_dir(request.task, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    object_category = request.task

    # 将耗时任务添加到后台任务队列
    background_tasks.add_task(run_navigation_task, save_dir, object_category)

    # 立即返回目录信息，不等待后台任务结束
    return {"obs_dir": obs_dir, "message": "Task started in background"}