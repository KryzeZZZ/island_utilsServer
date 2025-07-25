"""向量化服务

为对象关系和描述生成向量表示。
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

# 常量定义
VECTOR_DIM = 768  # sentence-transformers/paraphrase-mpnet-base-v2 输出维度

# 加载模型（全局单例）
model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

def _check_vector_dim(vector: List[float], source: str):
    """检查向量维度是否正确"""
    if len(vector) != VECTOR_DIM:
        raise ValueError(f"向量维度错误：期望 {VECTOR_DIM} 维，实际 {len(vector)} 维。来源：{source}")

def vectorize_relation(
    subject: str,
    predicate: str,
    object: str
) -> List[float]:
    """
    将三元组关系转换为向量
    
    参数:
        subject: 主体对象
        predicate: 关系
        object: 客体对象
    
    返回:
        {VECTOR_DIM}维向量
    """
    # 构造关系文本
    text = f"{subject} {predicate} {object}"
    
    # 生成向量
    vector = model.encode(text).tolist()
    _check_vector_dim(vector, f"关系：{text}")
    return vector

def vectorize_description(description: str) -> List[float]:
    """
    将对象描述转换为向量
    
    参数:
        description: 对象的描述文本
    
    返回:
        {VECTOR_DIM}维向量
    """
    vector = model.encode(description).tolist()
    _check_vector_dim(vector, f"描述：{description}")
    return vector

def batch_vectorize_relations(relations: List[Dict]) -> List[Dict]:
    """
    批量处理关系向量化
    
    参数:
        relations: [
            {
                "subject": str,
                "predicate": str,
                "object": str
            },
            ...
        ]
    
    返回:
        添加了{VECTOR_DIM}维向量的关系列表
    """
    result = []
    for rel in relations:
        vector = vectorize_relation(
            rel["subject"],
            rel["predicate"],
            rel["object"]
        )
        rel["vector"] = vector
        result.append(rel)
    return result

def batch_vectorize_descriptions(descriptions: List[str]) -> List[Dict]:
    """
    批量处理描述向量化
    
    参数:
        descriptions: ["描述1", "描述2", ...]
    
    返回:
        [{"description": str, "vector": List[float]}, ...]，向量维度为{VECTOR_DIM}
    """
    result = []
    for desc in descriptions:
        vector = vectorize_description(desc)
        result.append({
            "description": desc,
            "vector": vector
        })
    return result

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    注意：两个向量都必须是{VECTOR_DIM}维
    """
    _check_vector_dim(vec1, "相似度计算-向量1")
    _check_vector_dim(vec2, "相似度计算-向量2")
    
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# 测试用例
if __name__ == "__main__":
    print(f"\n使用 {VECTOR_DIM} 维向量进行编码\n")
    
    # 测试关系向量化
    test_relations = [
        {
            "subject": "魔法杖",
            "predicate": "增强",
            "object": "火系魔法"
        },
        {
            "subject": "铁锁",
            "predicate": "锁住",
            "object": "木门"
        }
    ]
    
    vectorized_relations = batch_vectorize_relations(test_relations)
    print("\n关系向量化示例：")
    for rel in vectorized_relations:
        print(f"\n关系：{rel['subject']} {rel['predicate']} {rel['object']}")
        print(f"向量维度：{len(rel['vector'])} (应为 {VECTOR_DIM})")
    
    # 测试描述向量化
    test_descriptions = [
        "一把精致的魔法杖，杖身镶嵌着红色的宝石",
        "一扇厚重的木门，上面有些年久失修的痕迹"
    ]
    
    vectorized_descriptions = batch_vectorize_descriptions(test_descriptions)
    print("\n描述向量化示例：")
    for item in vectorized_descriptions:
        print(f"\n描述：{item['description']}")
        print(f"向量维度：{len(item['vector'])} (应为 {VECTOR_DIM})")
    
    # 测试相似度计算
    sim = cosine_similarity(
        vectorized_relations[0]["vector"],
        vectorized_relations[1]["vector"]
    )
    print(f"\n两个关系的余弦相似度：{sim:.4f}") 