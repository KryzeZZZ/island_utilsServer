import requests


def generate_inner_scene(entry_term: str, external_objects: list, base_url: str, model="qwen3:32b") -> str:
    """
    根据入口词语和入口外部物体，生成入口“内部”的大场景描述（约 80 字）

    参数：
    - entry_term：入口词语（如“石门”、“洞口”）
    - external_objects：入口外部出现的物体列表

    返回：字符串，描述入口内部的场景
    """

    object_str = "、".join(external_objects)

    prompt = f"""
你是一个中文场景生成助手。

请根据以下信息，生成“入口内部”的中文大场景画面（约 80 个字），语言具有画面感。内容必须是从该入口进入后的空间，不要描述外部，不要解释。不要使用“这是一个...”等开场，no thinking, 直接生成场景内容。

入口词语：{entry_term}
入口外部物体：{object_str}

现在，请描述进入“{entry_term}”之后看到的场景：
"""

    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.9
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        output_text = data['choices'][0]['message']['content'].strip()
        return output_text
    except Exception as e:
        print("❌ 内部场景生成失败:", e)
        if response is not None:
            try:
                print("响应内容:", response.text)
            except:
                pass
        return "生成失败"


# ✅ 示例调用
if __name__ == "__main__":
    base_url = "https://d07241129-ollama-webui-qwen3v2-2962-tdyjpivz-11434.550c.cloud"
    entry_term = "石门"
    external_objects = ["石像", "藤蔓", "碎石", "铁链"]
    inner_scene = generate_inner_scene(entry_term, external_objects, base_url)
    print(f"入口：{entry_term}\n外部物体：{external_objects}\n入口内部场景：{inner_scene}")
