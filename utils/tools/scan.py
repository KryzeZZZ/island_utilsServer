import requests
import json
import re


def safe_parse_json(text: str) -> list:
    """
    尝试从返回中提取 JSON 列表
    """
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        json_match = re.search(r"\[\s*{.*?}\s*]", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print("⚠️ JSON 解析失败:", e)
    return []


def scan_objects_with_flags(text: str, base_url: str, model="qwen3:32b") -> list:
    """
    使用大模型分析中文文本中的物体，并标注以下属性：
    - refinable：是否是上级结构，可进一步细化
    - interactable：是否可以与之交互
    - is_entry：是否可能是“入口”类物体

    返回示例：
    [
        {"object": "门", "refinable": false, "interactable": true, "is_entry": true},
        {"object": "柜子", "refinable": true, "interactable": true, "is_entry": false}
    ]
    """

    prompt = f"""
请从下面的中文描述中提取所有“物体名词”（即 object），并判断以下三个属性：

1. refinable：是否是可细化的上级结构，如房子、桌子、床、书架等，设为 true。
2. interactable：是否可以与其交互，如可触碰、可拿起、可打开、可操作等，设为 true。
3. is_entry：是否是可作为通道或入口的物体，如门、窗、洞口、楼梯口等，设为 true。

返回 JSON 数组格式如下（不要添加解释、不要换行、不要加 markdown）：
[
  {{"object": "物体名1", "refinable": true, "interactable": false, "is_entry": false}},
  {{"object": "物体名2", "refinable": false, "interactable": true, "is_entry": true}},
  ...
]

描述文本如下：
{text}
"""

    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stop": None
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        output_text = data['choices'][0]['message']['content']
        return safe_parse_json(output_text)
    except Exception as e:
        print("❌ Ollama API 请求失败:", e)
        if response is not None:
            try:
                print("响应内容:", response.text)
            except:
                pass
        return []


# ✅ 示例调用
if __name__ == "__main__":
    base_url = "https://d07241129-ollama-webui-qwen3v2-2962-tdyjpivz-11434.550c.cloud"
    sample_text = "眼前有一座巨大的房子，房子里有一扇门，一个旧书架靠墙放着，上面放着花瓶。"
    results = scan_objects_with_flags(sample_text, base_url)
    for item in results:
        print(f"{item['object']} ｜ 可细化: {item['refinable']} ｜ 可交互: {item['interactable']} ｜ 是否入口: {item['is_entry']}")
