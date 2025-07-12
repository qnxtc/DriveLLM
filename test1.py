import os
from volcenginesdkarkruntime import Ark

# 直接设置您的方舟API Key
client = Ark(api_key="ca73283a-d4f5-4f15-9d6a-c117b712592d")

completion = client.chat.completions.create(
    # 使用有效的模型ID（移除尖括号或替换为其他模型）
    model="doubao-seed-1-6-250615",  # 或者替换为其他已知有效的模型ID
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(completion.choices[0].message)