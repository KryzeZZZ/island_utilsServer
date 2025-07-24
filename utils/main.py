from fastapi import FastAPI, Body
from pydantic import BaseModel
from services.motivate import extract_action_motives_ollama
from services.scan import scan_objects_with_flags
from services.moveenv import  generate_inner_scene
from services.render import refine_object_descriptions

app = FastAPI()

# 全局 Ollama 服务地址
BASE_URL = "https://d07241129-ollama-webui-qwen3v2-2962-tdyjpivz-11434.550c.cloud"

# 请求数据结构
class MotiveRequest(BaseModel):
    text: str

class ObjectScanRequest(BaseModel):
    text: str

class SceneRequest(BaseModel):
    entry_term: str
    external_objects: list[str]

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
