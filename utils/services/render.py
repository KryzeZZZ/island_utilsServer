import requests
import json

def refine_object_descriptions(objects: list, base_url: str, model: str = "deepseek-ai/DeepSeek-V3") -> dict:
    """
    根据输入的物体列表，使用 Ollama WebUI 模型生成每个物体的详细场景描述。

    参数:
        objects (list): 物体名称列表，例如 ["桌子", "沙发"]
        base_url (str): Ollama WebUI 的 Base URL
        model (str): 模型名称，默认 "qwen3:32b"

    返回:
        dict: 每个物体对应一个详细描述，例如 {"桌子": "桌子上堆满了杂志和文件"}
    """

    results = {}

    for obj in objects:
        prompt = f"""请详细描述“{obj}”在一个真实室内场景中的样子和摆放情况。不超过50字。不要思考，不要输出 <think>。直接输出描述。"""

        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 30,
            "enable_thinking": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization":  "Bearer sk-tbcaipmckbmbxkjpzmwgfutjtlgsnqecdmlvhlelinckvwok"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            description = data['choices'][0]['message']['content'].strip()

            # 清理不需要的内容
            if "<think>" in description:
                description = description.split("</think>")[-1].strip()
            results[obj] = description

        except Exception as e:
            print(f"⚠️ 无法获取 [{obj}] 的描述:", e)
            if response is not None:
                print("响应内容:", response.text)
            results[obj] = "（描述失败）"

    return results


# ✅ 示例运行
if __name__ == "__main__":
    base_url = "https://d07241129-ollama-webui-qwen3v2-2962-tdyjpivz-11434.550c.cloud"
    initial_objects = [""]
    refined = refine_object_descriptions(initial_objects, base_url)
    for obj, desc in refined.items():
        print(f"{obj}：{desc}")
