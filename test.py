# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2025-07-10
# @Author  : white
# @FileName: test.py
# @Software: PyCharm
# **************************************
def test_gpt():
    import requests

    # 设置 API Key（请替换为你自己的 Key，注意保密）
    api_key = "sk-GSP9lA0Ha8aeee816804T3BLBkFJ398e1f4dd018433592E9"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    params = {
        "model": "gpt-3.5-turbo-1106",  # 可根据需要切换为 gpt-3.5-turbo 等
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ]
    }

    try:
        response = requests.post(
            "https://api.ohmygpt.com/v1/chat/completions",  # 正确的 API 地址
            headers=headers,
            json=params,
            timeout=20  # 设置超时时间
        )

        if response.status_code == 200:
            res = response.json()
            res_content = res['choices'][0]['message']['content']
            print("GPT response:", res_content)
        else:
            print("Request failed with status code:", response.status_code)
            print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

# 执行函数
test_gpt()
#确定模型的调用方式
