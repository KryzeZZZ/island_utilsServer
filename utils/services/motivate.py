import requests
import re

# 从 Ollama 返回中提取 <think> 后的实际内容
def extract_after_think_block(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()

# 你自己的动机词表
KNOWN_MOTIVES = {"物品交互", "转移", "自身行为"}

# 通过 Ollama 总结一个动作短语的动机
def summarize_action_phrase_ollama(phrase: str, base_url: str, model_name="deepseek-ai/DeepSeek-V3") -> str:
    prompt = f"""请从{KNOWN_MOTIVES}中挑选合适的词作为动机,只返回词，不要新建,有物品被使用了就是物品交互 no thinking：
动作：{phrase}
总结（直接返回，不要加上动机:）："""

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9
    }

    headers = {"Content-Type": "application/json", "Authorization":  "Bearer sk-tbcaipmckbmbxkjpzmwgfutjtlgsnqecdmlvhlelinckvwok"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(data)
        # 获取返回的字符串内容
        output_text = data['choices'][0]['message']['content'].strip()
        # 提取 <think> 后的部分（如果存在）
        output_text = extract_after_think_block(output_text)

        # 去除前缀“答案：”和空行等
        output_text = re.sub(r"^答案[:：]?", "", output_text).strip()
        output_text = output_text.replace("\n", " ").strip()

        # 提取第一个合理的动机词
        candidates = re.split(r"[，。,.，\s]+", output_text)
        for c in candidates:
            if c.strip():
                return c.strip()

    except Exception as e:
        print("⚠️ 动机提取失败:", e)
        try:
            print("响应:", response.text)
        except:
            pass

    return "未知动机"

# 主动机提取器
def extract_action_motives_ollama(text: str, base_url: str, known_motives=KNOWN_MOTIVES) -> list:
    action_phrases = re.split(r"[，、]", text)
    results = []

    for phrase in action_phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        matched = None
        for motive in known_motives:
            if motive in phrase:
                matched = motive
                break

        if matched is None:
            summary = summarize_action_phrase_ollama(phrase, base_url)
            matched = summary

        results.append({
            "action_phrase": phrase,
            "motive": matched
        })

    return results

# ✅ 示例
if __name__ == "__main__":
    base_url = "https://d07241129-ollama-webui-qwen3v2-2962-tdyjpivz-11434.550c.cloud"
    text = "他拿起刀往我这砍，我往后一躲"
    motives = extract_action_motives_ollama(text, base_url)
    for m in motives:
        print(f"动作: {m['action_phrase']} ｜ 动机: {m['motive']}")
