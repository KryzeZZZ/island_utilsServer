"""掷骰子服务

根据角色背景、目标关系和环境关系网络，判定动作是否成功。
"""

import random
import requests
from typing import TypeVar, List, Dict


def roll_for_action(
    persona: str,
    target_relation: Dict[str, str],
    env_relations: List[Dict[str, str]],
    base_url: str,
    model: str = "deepseek-ai/DeepSeek-V3"
) -> Dict:
    """
    根据角色背景和关系网络进行动作判定
    
    返回:
        {
            "success": bool,      # 是否成功
            "reason": str,        # 成功/失败原因
            "roll": int,         # 实际掷骰结果 (1-100)
            "difficulty": int,    # 难度等级 (1-100)
            "outcome": str       # AI生成的结果描述
        }
    """
    
    # 第一步：判定难度和掷骰
    difficulty_prompt = f"""
你是一个 TRPG 游戏主持人。请根据以下信息设定一个动作难度等级（1-100），并给出原因：

# 角色背景
{persona}

# 目标动作
{target_relation['subject']} 试图 {target_relation['predicate']} {target_relation['object']}

# 环境关系
{chr(10).join([f"{r['subject']} 与 {r['object']} 的关系是：{r['predicate']}" for r in env_relations])}

请按以下格式输出（不要其他内容）：
难度等级：[数字1-100]
原因：[难度判定依据，限50字]
"""

    # 调用 API 获取难度等级
    url = f"{base_url}/v1/chat/completions"
    difficulty_payload = {
        "model": model,
        "messages": [{"role": "user", "content": difficulty_prompt}],
        "temperature": 0.3,
        "max_tokens": 150
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
        lines = output.split("\n")
        difficulty = int(lines[0].split("：")[1].strip())
        reason = lines[1].split("：")[1].strip()
        
        # 掷骰子 (1-100)
        roll = random.randint(1, 100)
        success = roll <= difficulty

        # 第二步：生成结果描述
        outcome_prompt = f"""
你是一个 TRPG 游戏主持人。请根据以下信息生成一段生动的结果描述（限100字）：

# 角色背景
{persona}

# 目标动作
{target_relation['subject']} 试图 {target_relation['predicate']} {target_relation['object']}

# 环境关系
{chr(10).join([f"{r['subject']} 与 {r['object']} 的关系是：{r['predicate']}" for r in env_relations])}

# 判定结果
难度等级：{difficulty}
掷骰结果：{roll}
最终结果：{'成功' if success else '失败'}
原因：{reason}

请直接输出描述文本，不要其他内容。描述要有画面感，突出动作过程和结果。
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
            result_reason = f"✅ {reason}，掷骰 {roll} ≤ 难度 {difficulty}"
        else:
            result_reason = f"❌ {reason}，掷骰 {roll} > 难度 {difficulty}"
        
        return {
            "success": success,
            "reason": result_reason,
            "roll": roll,
            "difficulty": difficulty,
            "outcome": outcome
        }

    except Exception as e:
        print(f"⚠️ 动作判定失败: {str(e)}")
        return {
            "success": False,
            "reason": f"⚠️ 判定出错: {str(e)}",
            "roll": 0,
            "difficulty": 100,
            "outcome": "发生了一些意外，无法确定具体结果。"
        }


# 测试用例
if __name__ == "__main__":
    test_persona = "一位经验丰富的盗贼，擅长开锁和潜行。"
    test_relation = {
        "subject": "盗贼",
        "predicate": "撬开",
        "object": "生锈的铁锁"
    }
    test_env = [
        {
            "subject": "铁锁",
            "predicate": "已经生锈多年",
            "object": "木门"
        },
        {
            "subject": "盗贼",
            "predicate": "携带",
            "object": "专业开锁工具"
        }
    ]
    
    result = roll_for_action(
        test_persona,
        test_relation,
        test_env,
        "http://localhost:11434"
    )
    
    print("\n判定结果：")
    print(result["reason"])
    print(f"掷骰: {result['roll']}")
    print(f"难度: {result['difficulty']}") 