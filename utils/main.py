from fastapi import FastAPI, Body
from pydantic import BaseModel
from services.motivate import extract_action_motives_ollama
from services.scan import scan_objects_with_flags
from services.moveenv import  generate_inner_scene
from services.render import refine_object_descriptions
from services.divide import extract_relations_ollama_webui
from services.dice import roll_for_action
from services.user_dice import roll_for_user_action
from services.vectorize import batch_vectorize_relations, batch_vectorize_attributes
from typing import List, Dict
import os

app = FastAPI()

# 全局 Ollama 服务地址
BASE_URL = "https://api.siliconflow.cn"

# 请求数据结构
class MotiveRequest(BaseModel):
    text: str

class ObjectScanRequest(BaseModel):
    text: str

class SceneRequest(BaseModel):
    entry_term: str
    external_objects: list[str]

class DiceRollRequest(BaseModel):
    """掷骰子请求结构"""
    persona: str
    target_relation: Dict[str, str]  # {"subject": "", "predicate": "", "object": ""}
    env_relations: List[Dict[str, str]]  # [{"subject": "", "predicate": "", "object": ""}, ...]

class UserActionRequest(BaseModel):
    """用户动作判定请求结构"""
    persona: str  # 用户背景
    action: str   # 要执行的动作
    related_objects: List[Dict[str, str]]  # [{"object": "物品描述", "relation": "与用户的关系"}, ...]

class VectorizeRelationsRequest(BaseModel):
    """关系向量化请求"""
    relations: List[Dict[str, str]]  # [{"subject": "", "predicate": "", "object": "", "context": ""}, ...]

class VectorizeAttributesRequest(BaseModel):
    """属性向量化请求"""
    attributes: List[Dict]  # [{"object_name": "", "attribute_name": "", "attribute_value": "", "context": ""}, ...]

class VectorSearchRequest(BaseModel):
    """向量搜索请求"""
    query_vector: List[float]
    threshold: float = 0.7
    limit: int = 10

# 接口1：动作动机识别
@app.post("/extract_motives")
def extract_motives(req: MotiveRequest):
    return extract_action_motives_ollama(req.text, BASE_URL)

# 接口2：物体扫描 + 三维判断
@app.post("/scan_objects")
def scan_objects(req: ObjectScanRequest):
    return scan_objects_with_flags(req.text, BASE_URL)

# 接口3：生成物品
@app.post("/generate/detail_obj")
def obj_details(req: SceneRequest):
    return {"scene": refine_object_descriptions(req.objects, BASE_URL)}

# 接口4：生成入口内部场景
@app.post("/generate/inner_scene")
def inner_scene(req: SceneRequest):
    return {"scene": generate_inner_scene(req.entry_term, req.external_objects, BASE_URL)}

#接口5: 获取一段话内的Object关系
@app.post("/relationship")
def relationship(req: MotiveRequest):
    return {extract_relations_ollama_webui(req.text, BASE_URL)}

# 接口6: 动作判定掷骰
@app.post("/roll_dice")
def roll_dice(req: DiceRollRequest):
    """根据角色背景和关系网络进行动作判定"""
    result = roll_for_action(
        persona=req.persona,
        target_relation=req.target_relation,
        env_relations=req.env_relations,
        base_url=BASE_URL
    )
    return result

# 接口7: 用户动作判定
@app.post("/roll_user_action")
def roll_user_action(req: UserActionRequest):
    """基于用户背景和相关物品判定行动成功率"""
    result = roll_for_user_action(
        persona=req.persona,
        action=req.action,
        related_objects=req.related_objects,
        base_url=BASE_URL
    )
    return result

# 接口8: 关系向量化
@app.post("/vectorize/relations")
def vectorize_relations(req: VectorizeRelationsRequest):
    """将关系转换为向量表示"""
    return batch_vectorize_relations(req.relations)

# 接口9: 属性向量化
@app.post("/vectorize/attributes")
def vectorize_attributes(req: VectorizeAttributesRequest):
    """将属性转换为向量表示"""
    return batch_vectorize_attributes(req.attributes)
