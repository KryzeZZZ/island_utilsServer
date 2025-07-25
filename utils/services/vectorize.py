"""向量化服务

为对象关系和属性生成向量表示，用于 Neo4j 向量搜索。
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import numpy as np

# 加载模型（全局单例）
model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

def vectorize_relation(
    subject: str,
    predicate: str,
    object: str,
    context: Optional[str] = None
) -> List[float]:
    """
    将三元组关系转换为向量
    
    参数:
        subject: 主体
        predicate: 谓语/关系
        object: 客体
        context: 可选的上下文信息
    
    返回:
        768维向量
    """
    # 构造关系文本
    if context:
        text = f"在{context}中，{subject} {predicate} {object}"
    else:
        text = f"{subject} {predicate} {object}"
    
    # 生成向量
    return model.encode(text).tolist()

def vectorize_attribute(
    object_name: str,
    attribute_name: str,
    attribute_value: Union[str, int, float, bool],
    context: Optional[str] = None
) -> List[float]:
    """
    将对象属性转换为向量
    
    参数:
        object_name: 对象名称
        attribute_name: 属性名
        attribute_value: 属性值
        context: 可选的上下文信息
    
    返回:
        768维向量
    """
    # 构造属性文本
    if context:
        text = f"在{context}中，{object_name}的{attribute_name}是{str(attribute_value)}"
    else:
        text = f"{object_name}的{attribute_name}是{str(attribute_value)}"
    
    # 生成向量
    return model.encode(text).tolist()

def batch_vectorize_relations(relations: List[Dict]) -> List[Dict]:
    """
    批量处理关系向量化
    
    参数:
        relations: [
            {
                "subject": str,
                "predicate": str,
                "object": str,
                "context": str (optional)
            },
            ...
        ]
    
    返回:
        添加了向量的关系列表
    """
    result = []
    for rel in relations:
        vector = vectorize_relation(
            rel["subject"],
            rel["predicate"],
            rel["object"],
            rel.get("context")
        )
        rel["vector"] = vector
        result.append(rel)
    return result

def batch_vectorize_attributes(attributes: List[Dict]) -> List[Dict]:
    """
    批量处理属性向量化
    
    参数:
        attributes: [
            {
                "object_name": str,
                "attribute_name": str,
                "attribute_value": Union[str, int, float, bool],
                "context": str (optional)
            },
            ...
        ]
    
    返回:
        添加了向量的属性列表
    """
    result = []
    for attr in attributes:
        vector = vectorize_attribute(
            attr["object_name"],
            attr["attribute_name"],
            attr["attribute_value"],
            attr.get("context")
        )
        attr["vector"] = vector
        result.append(attr)
    return result

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# 测试用例
if __name__ == "__main__":
    # 测试关系向量化
    test_relations = [
        {
            "subject": "魔法杖",
            "predicate": "增强",
            "object": "火系魔法",
            "context": "战斗场景"
        },
        {
            "subject": "铁锁",
            "predicate": "锁住",
            "object": "木门",
            "context": "地牢入口"
        }
    ]
    
    vectorized_relations = batch_vectorize_relations(test_relations)
    print("\n关系向量化示例：")
    for rel in vectorized_relations:
        print(f"\n关系：{rel['subject']} {rel['predicate']} {rel['object']}")
        print(f"向量维度：{len(rel['vector'])}")
        
    # 测试属性向量化
    test_attributes = [
        {
            "object_name": "魔法杖",
            "attribute_name": "品质",
            "attribute_value": "精良",
            "context": "装备属性"
        },
        {
            "object_name": "木门",
            "attribute_name": "状态",
            "attribute_value": "破损",
            "context": "物品状态"
        }
    ]
    
    vectorized_attributes = batch_vectorize_attributes(test_attributes)
    print("\n属性向量化示例：")
    for attr in vectorized_attributes:
        print(f"\n属性：{attr['object_name']}的{attr['attribute_name']}是{attr['attribute_value']}")
        print(f"向量维度：{len(attr['vector'])}")
    
    # 测试相似度计算
    sim = cosine_similarity(
        vectorized_relations[0]["vector"],
        vectorized_relations[1]["vector"]
    )
    print(f"\n两个关系的余弦相似度：{sim:.4f}") 