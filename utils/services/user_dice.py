"""用户动作判定服务

基于用户背景和相关物品网络，判定行动是否成功。
"""

import random
import requests
from typing import List, Dict



def roll_for_user_action(
    persona: str,
    action: str,
    related_objects: List[Dict[str, str]],
    base_url: str,
    model: str = "deepseek-ai/DeepSeek-V3"
) -> Dict:
    """
    基于用户背景和相关物品判定行动成功率
    
    返回:
        {
            "success": bool,      # 是否成功
            "reason": str,        # 成功/失败原因
            "roll": int,         # 实际掷骰结果 (1-100)
            "difficulty": int,    # 基础难度等级 (1-100)
            "modifiers": list[str], # 影响因素列表
            "outcome": str       # AI生成的结果描述
        }
    """
    
    # 第一步：判定难度和影响因素
    difficulty_prompt = f"""
你是一个 TRPG 游戏主持人。请分析玩家背景和相关物品，设定一个动作难度等级（1-100），并列出所有影响因素：

# 玩家背景
{persona}

# 目标动作
{action}

# 相关物品
{chr(10).join([f"- {obj['object']}: {obj['relation']}" for obj in related_objects])}

请按以下格式输出（不要其他内容）：
基础难度：[数字1-100]
影响因素：
[因素1，+/-效果]
[因素2，+/-效果]
...
总结：[最终难度判定依据，限50字]
"""

    # 调用 API 获取难度分析
    url = f"{base_url}/v1/chat/completions"
    difficulty_payload = {
        "model": model,
        "messages": [{"role": "user", "content": difficulty_prompt}],
        "temperature": 0.3,
        "max_tokens": 250
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-tbcaipmckbmbxkjpzmwgfutjtlgsnqecdmlvhlelinckvwok"
    }

    try:
        response = requests.post(url, headers=headers, json=difficulty_payload, timeout=30)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"].strip()
        
        # 解析输出
        sections = output.split("\n")
        base_difficulty = int(sections[0].split("：")[1].strip())
        
        # 提取影响因素
        modifiers = []
        final_difficulty = base_difficulty
        
        in_modifiers = False
        for line in sections[1:]:
            if line.startswith("影响因素"):
                in_modifiers = True
                continue
            elif line.startswith("总结"):
                reason = line.split("：")[1].strip()
                break
            elif in_modifiers and line.strip():
                modifiers.append(line.strip())
                # 尝试从影响因素中提取数值调整
                try:
                    if "+" in line:
                        final_difficulty += int(line.split("+")[1].split("]")[0])
                    elif "-" in line:
                        final_difficulty -= int(line.split("-")[1].split("]")[0])
                except:
                    pass
        
        # 确保最终难度在1-100范围内
        final_difficulty = max(1, min(100, final_difficulty))
        
        # 掷骰子 (1-100)
        roll = random.randint(1, 100)
        success = roll <= final_difficulty

        # 第二步：生成结果描述
        outcome_prompt = f"""
你是一个 TRPG 游戏主持人。请根据以下信息生成一段生动的结果描述（限100字）：

# 玩家背景
{persona}

# 目标动作
{action}

# 相关物品
{chr(10).join([f"- {obj['object']}: {obj['relation']}" for obj in related_objects])}

# 判定结果
基础难度：{base_difficulty}
影响因素：
{chr(10).join(modifiers)}
最终难度：{final_difficulty}
掷骰结果：{roll}
最终结果：{'成功' if success else '失败'}

请直接输出描述文本，不要其他内容。描述要有画面感，突出动作过程、物品效果和最终结果。
"""

        outcome_payload = {
            "model": model,
            "messages": [{"role": "user", "content": outcome_prompt}],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        response = requests.post(url, headers=headers, json=outcome_payload, timeout=30)
        response.raise_for_status()
        outcome = response.json()["choices"][0]["message"]["content"].strip()
        
        # 生成结果描述
        if success:
            result_reason = f"✅ {reason}，掷骰 {roll} ≤ 难度 {final_difficulty}"
        else:
            result_reason = f"❌ {reason}，掷骰 {roll} > 难度 {final_difficulty}"
        
        return {
            "success": success,
            "reason": result_reason,
            "roll": roll,
            "difficulty": final_difficulty,
            "base_difficulty": base_difficulty,
            "modifiers": modifiers,
            "outcome": outcome
        }

    except Exception as e:
        print(f"⚠️ 动作判定失败: {str(e)}")
        return {
            "success": False,
            "reason": f"⚠️ 判定出错: {str(e)}",
            "roll": 0,
            "difficulty": 100,
            "modifiers": [],
            "outcome": "发生了一些意外，无法确定具体结果。"
        }


# 测试用例
if __name__ == "__main__":
    test_persona = "一位年轻的魔法学徒，擅长火系魔法，但经验尚浅。"
    test_action = "试图用火球术击中20米外的移动靶子"
    test_objects = [
        {
            "object": "精制魔法杖",
            "relation": "最近刚刚获得的趁手武器"
        },
        {
            "object": "魔法水晶",
            "relation": "佩戴在胸前，可以增强火系魔法"
        },
        {
            "object": "破损的眼镜",
            "relation": "视力不太好，影响瞄准"
        }
    ]
    
    result = roll_for_user_action(
        test_persona,
        test_action,
        test_objects,
        "http://localhost:11434"
    )
    
    print("\n判定结果：")
    print(result["reason"])
    print(f"基础难度: {result['base_difficulty']}")
    print("\n影响因素：")
    for mod in result["modifiers"]:
        print(f"- {mod}")
    print(f"\n最终难度: {result['difficulty']}")
    print(f"掷骰: {result['roll']}") 