import os
import time
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from typing import Optional, List, Mapping, Any, Dict, Awaitable
import asyncio

# 导入火山方舟官方SDK
from volcenginesdkarkruntime import Ark
# 导入LangChain聊天模型基类及消息类型
from langchain.chat_models.base import BaseChatModel, ChatResult, ChatGeneration
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

# 禁用SSL验证（仅测试用，生产环境建议启用）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 手动注册highway-v0环境
from gymnasium.envs.registration import register
register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)


# 定义火山方舟Doubao模型的聊天模型包装器（支持消息列表输入）
class VolcanoDoubaoChatModel(BaseChatModel):
    api_key: str
    model_name: str  # 模型ID，如'doubao-seed-1-6-250615'

    @property
    def _llm_type(self) -> str:
        return "volcano-doubao-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,** kwargs: Any,
    ) -> ChatResult:
        """核心同步方法：处理消息列表并调用火山方舟API"""
        # 1. 将LangChain消息转换为火山方舟API格式
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            else:
                # 忽略不支持的消息类型
                continue

        # 2. 调用火山方舟Doubao模型
        client = Ark(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=0.0,  # 保持决策确定性
            max_tokens=2048,  # 增大token限制，避免序列过长
            stop=stop  # 传递停止词（如果有）
        )

        # 3. 转换API响应为LangChain所需的ChatResult格式
        generation = ChatGeneration(
            message=AIMessage(content=completion.choices[0].message.content)
        )
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成方法：基于同步方法封装（满足抽象类要求）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # 使用默认线程池
            self._generate,  # 调用同步方法
            messages, stop, run_manager,** kwargs
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """用于标识模型的参数（调试用）"""
        return {
            "model_name": self.model_name,
            "api_key": self.api_key[:4] + "****"  # 隐藏部分API密钥
        }


# 加载火山方舟配置（指定UTF-8编码避免解码错误）
VOLCANO_CONFIG = yaml.load(open('config.yaml', encoding='utf-8'), Loader=yaml.FullLoader)

# 初始化Doubao聊天模型（使用重构后的ChatModel）
llm = VolcanoDoubaoChatModel(
    api_key=VOLCANO_CONFIG['VOLCANO_API_KEY'],
    model_name=VOLCANO_CONFIG['VOLCANO_MODEL']
)

# 基础设置
vehicleCount = 15

# 环境配置
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

# 环境初始化流程
env = gym.make('highway-v0', render_mode="rgb_array")
env = env.unwrapped
env.configure(config)
obs, info = env.reset()
# 初始化帧号
frame = 0

# 视频记录：显式配置录制参数并强制启动
env = RecordVideo(
    env,
    video_folder='./results-video',  # 视频保存路径
    name_prefix="highwayv0",         # 视频文件前缀
    episode_trigger=lambda x: True,  # 强制触发录制（每帧都录制）
    disable_logger=True              # 禁用冗余日志输出
)
# 手动启动录制，传入视频文件名（使用帧号确保唯一性）
video_name = f"{env.name_prefix}_{frame:06d}"  # 例如：highwayv0_000000
env.start_recording(video_name=video_name)

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
        # 更新场景车辆信息
        sce.upateVehicles(obs, frame)
        # 驱动代理决策
        DA.agentRun(output)
        # 解析决策结果
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)
        # 渲染画面（自动被RecordVideo捕获）
        env.render()
        # 执行动作并获取新状态
        obs, reward, terminated, truncated, info = env.step(output["action_id"])
        done = terminated or truncated
        # 打印决策结果
        print(output)
        frame += 1
finally:
    # 停止录制并关闭环境
    env.stop_recording()
    env.close()