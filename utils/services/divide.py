import requests
import json
import re

def safe_parse_json(text: str) -> list:
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        json_match = re.search(r"\[\s*{.*}\s*]", text, re.DOTALL)
        if json_match:
            clean = json_match.group(0)
            return json.loads(clean)
    except Exception as e:
        print("⚠️ JSON 解析失败:", e)
    return []

def extract_relations_ollama_webui(text: str, base_url: str) -> list:
    prompt = f"""
请从下面的文本中抽取出所有明确的三元组关系（subject, relation, object）：
no thinkning
要求：
- 输出格式为 JSON 数组，例如：
  [
    {{"subject": "A", "relation": "B", "object": "C"}}
  ]
- 如果没有可提取的关系，请返回空数组 []
- 不要解释说明，也不要包含 markdown 或任何前后缀

文本如下：
{text}
"""

    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": "deepseek-ai/DeepSeek-V3",  # 确认你用的模型名
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 30,
        "enable_thinking": False
    }

    headers = {
        "Content-Type": "application/json", "Authorization":  "Bearer sk-tbcaipmckbmbxkjpzmwgfutjtlgsnqecdmlvhlelinckvwok"
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # 解析返回的文本，路径可能是 data['choices'][0]['message']['content']
        output_text = data['choices'][0]['message']['content']
        triplets = safe_parse_json(output_text)
        return triplets
    except Exception as e:
        print("⚠️ Ollama WebUI 调用失败:", e)
        return []

if __name__ == "__main__":
    base_url = "https://d07241129-ollama-webui-qwen3v2-2962-tdyjpivz-11434.550c.cloud"
    test_text = "我跳起来，拿起一瓶奶茶"
    triplets = extract_relations_ollama_webui(test_text, base_url)
    if not triplets:
        print("⚠️ 未提取到三元组。")
    for t in triplets:
        print(f"{t['subject']} --{t['relation']}--> {t['object']}")
