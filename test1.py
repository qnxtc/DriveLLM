# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2025-07-10
# @Author  : white
# @FileName: test.py
# @Software: PyCharm
# **************************************
def test_gpt():
    import requests
    import os

    # 禁用代理
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

    # 设置 API Key
    api_key = "sk-..."  # 注意保密

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    params = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ]
    }

    try:
        response = requests.post(
            "https://api.ohmygpt.com/v1/chat/completions",
            headers=headers,
            json=params,
            timeout=20
        )

        if response.status_code == 200:
            res = response.json()
            print("GPT response:", res['choices'][0]['message']['content'])
        else:
            print("Request failed:", response.status_code)
            print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

# 执行
test_gpt()
