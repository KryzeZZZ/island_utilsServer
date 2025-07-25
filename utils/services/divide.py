"""文本向量化和关系提取服务"""

import requests
from typing import List, Optional, Dict, Union
from sentence_transformers import SentenceTransformer
import numpy as np

# 初始化 Sentence Transformer 模型
_model = None

def get_model() -> SentenceTransformer:
    """获取或初始化 Sentence Transformer 模型"""
    global _model
    if _model is None:
        _model = SentenceTransformer('paraphrase-mpnet-base-v2')
    return _model

def generate_embedding_768(text: str) -> Optional[List[float]]:
    """生成文本的768维向量表示
    
    参数:
        text: 输入文本
        
    返回:
        768维向量，如果失败则返回 None
    """
    try:
        # 获取模型
        model = get_model()
        
        # 生成向量
        embedding = model.encode(
            text,
            convert_to_tensor=False,  # 返回numpy数组
            normalize_embeddings=True  # 标准化向量
        )
        
        # 确保维度正确
        if embedding.shape[0] != 768:
            print(f"⚠️ 向量维度错误: {embedding.shape[0]}")
            return None
        
        # 转换为列表并返回
        return embedding.tolist()
        
    except Exception as e:
        print(f"⚠️ 生成向量失败: {e}")
        return None

def extract_relations_ollama_webui(
    text: str,
    base_url: str
) -> List[Dict[str, str]]:
    """提取文本中的关系三元组
    
    参数:
        text: 输入文本
        base_url: Ollama WebUI API地址
        
    返回:
        关系三元组列表：[{"subject": str, "predicate": str, "object": str}]
    """
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "prompt": f"从以下文本中提取所有的关系三元组（主语-谓语-宾语）。\n\n文本：{text}\n\n请按以下格式返回（每行一个三元组）：\n主语|谓语|宾语",
                "system": "你是一个关系提取器。请从文本中提取所有可能的关系三元组，每个三元组包含主语、谓语和宾语。",
                "max_tokens": 200,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            # 解析返回的三元组
            triples = []
            for line in response.json()["text"].strip().split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) == 3:
                        triples.append({
                            "subject": parts[0].strip(),
                            "predicate": parts[1].strip(),
                            "object": parts[2].strip()
                        })
            return triples
            
        return []
        
    except Exception as e:
        print(f"⚠️ 提取关系失败: {e}")
        return []

# 测试用例
if __name__ == "__main__":
    # 测试向量生成
    test_text = "一个神秘的洞穴入口，周围长满了荧光蘑菇。"
    vector = generate_embedding_768(test_text)
    if vector:
        print(f"\n文本向量化测试:")
        print(f"- 输入: {test_text}")
        print(f"- 向量维度: {len(vector)}")
        print(f"- 向量前5个值: {vector[:5]}")
    
    # 测试关系提取
    base_url = "http://localhost:11434"
    test_text = "洞穴入口很黑暗，墙壁上爬满了发光的蘑菇，一个木制的门半掩着。"
    relations = extract_relations_ollama_webui(test_text, base_url)
    
    print("\n关系提取测试:")
    print(f"输入: {test_text}")
    print("提取的关系:")
    for rel in relations:
        print(f"- {rel['subject']} -> {rel['predicate']} -> {rel['object']}")
