import os
import time
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from typing import Optional, List, Mapping, Any

# 导入火山方舟官方SDK
from volcenginesdkarkruntime import Ark
# 导入LangChain基础类
from langchain.llms.base import LLM

# 禁用SSL验证（可选，仅测试用）
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 手动注册highway-v0环境
from gymnasium.envs.registration import register

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)


# 定义火山方舟Doubao模型的LLM包装器（基于官方SDK）
class VolcanoDoubaoLLM(LLM):
    api_key: str
    model_name: str  # 模型ID，如'doubao-seed-1-6-250615'

    @property
    def _llm_type(self) -> str:
        return "volcano-doubao"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 初始化火山方舟客户端
        client = Ark(api_key=self.api_key)

        # 调用Doubao模型
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 保持决策确定性
            max_tokens=1024
        )

        # 返回模型响应内容
        return completion.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name
        }


# 加载火山方舟配置（指定UTF-8编码避免解码错误）
VOLCANO_CONFIG = yaml.load(open('config.yaml', encoding='utf-8'), Loader=yaml.FullLoader)

# 初始化LLM模型（Doubao-Seed-1.6）
llm = VolcanoDoubaoLLM(
    api_key=VOLCANO_CONFIG['VOLCANO_API_KEY'],
    model_name=VOLCANO_CONFIG['VOLCANO_MODEL']
)

# 基础设置
vehicleCount = 15

# 环境配置
config = {
    "observation"      : {
        "type"          : "Kinematics",
        "features"      : ["presence", "x", "y", "vx", "vy"],
        "absolute"      : True,
        "normalize"     : False,
        "vehicles_count": vehicleCount,
        "see_behind"    : True,
    },
    "action"           : {
        "type"         : "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration"         : 40,
    "vehicles_density" : 2,
    "show_trajectories": True,
    "render_agent"     : True,
}

# 环境初始化流程
env = gym.make('highway-v0', render_mode="rgb_array")
env = env.unwrapped
env.configure(config)
obs, info = env.reset()

# 视频记录
env = RecordVideo(
    env, './results-video',
    name_prefix=f"highwayv0"
)
env.unwrapped.set_record_video_wrapper(env)
env.render()

# 场景和驾驶代理设置
if not os.path.exists('results-db/'):
    os.mkdir('results-db')
database = f"results-db/highwayv0.db"
from scenario.scenario import Scenario  # 导入Scenario类

sce = Scenario(vehicleCount, database)

# 工具模型初始化
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)

toolModels = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]

from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser

DA = DriverAgent(llm, toolModels, sce, verbose=True)
outputParser = OutputParser(sce, llm)
output = None
done = truncated = False
frame = 0

try:
    while not (done or truncated):
        sce.upateVehicles(obs, frame)
        DA.agentRun(output)
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)
        env.render()
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()
        obs, reward, terminated, truncated, info = env.step(output["action_id"])
        done = terminated or truncated
        print(output)
        frame += 1
finally:
    env.close()