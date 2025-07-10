import os
# 替换为你的代理软件的实际端口（例如7890、8080等）
PROXY_PORT = "5131"  # 重点：这里必须与代理软件的端口一致！
os.environ["http_proxy"] = "http://212.50.248.243"  # 替换为你的代理
os.environ["https_proxy"] = "http://212.50.248.243"

# # 第二步：禁用 SSL 验证（可选，仅测试用）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
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

# 手动注册 highway-v0 环境（确保环境能被识别）
from gymnasium.envs.registration import register
register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

# 加载OpenAI配置
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

# 初始化LLM模型
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
    os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
    os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
    llm = AzureChatOpenAI(
        deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
        temperature=0,
        max_tokens=1024,
        request_timeout=60
    )
elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-1106',  # 确保模型支持8k+上下文
        max_tokens=1024,
        request_timeout=60
    )

# 基础设置
vehicleCount = 15

# 环境配置（使用嵌套格式，适配原始环境的configure方法）
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "vehicles_count": vehicleCount,
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": 40,
    "vehicles_density": 2,
    "show_trajectories": True,
    "render_agent": True,
}

# 环境初始化流程（关键修改）
# 1. 创建环境（不传入配置参数）
env = gym.make('highway-v0', render_mode="rgb_array")
# 2. 获取原始环境（解除gymnasium包装器）
env = env.unwrapped
# 3. 应用配置
env.configure(config)
# 4. 重置环境使配置生效
obs, info = env.reset()

# 视频记录（在配置后包装，避免影响原始环境）
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
sce = Scenario(vehicleCount, database)

# 工具模型初始化（使用配置后的环境）
toolModels = [
    getAvailableActions(env),  # 传入配置后的env
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]

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
        # 执行动作（注意：新版gymnasium的step返回值为(obs, reward, terminated, truncated, info)）
        obs, reward, terminated, truncated, info = env.step(output["action_id"])
        done = terminated or truncated  # 兼容旧版done逻辑
        print(output)
        frame += 1
finally:
    env.close()