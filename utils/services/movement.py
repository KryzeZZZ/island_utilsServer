"""移动服务

处理用户移动指令，支持目的地查找和方向移动。
"""

from typing import Dict, List, Optional, Tuple
import math
import requests
from .graphdb import GraphDB
from .divide import generate_embedding_768

class MovementService:
    def __init__(self, graph: GraphDB, base_url: str):
        """
        初始化移动服务
        
        参数:
            graph: GraphDB实例
            base_url: LLM服务基础URL
        """
        self.graph = graph
        self.base_url = base_url
    
    def extract_destination(self, command: str) -> Optional[str]:
        """从指令中提取目的地描述"""
        try:
            prompt = f"""
你是一个移动指令分析器。请从以下指令中提取目的地描述。
如果指令中没有明确的目的地（比如"向前走"），返回空字符串。

指令：{command}

请直接返回提取的目的地或空字符串（不要其他任何内容）。
"""
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 50
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-tbcaipmckbmbxkjpzmwgfutjtlgsnqecdmlvhlelinckvwok"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                destination = response.json()["choices"][0]["message"]["content"].strip()
                return destination if destination else None
            return None
            
        except Exception as e:
            print(f"⚠️ 提取目的地失败: {e}")
            return None
    
    def find_destination_scene(self, description: str) -> Optional[Dict]:
        """使用向量搜索查找目的地场景"""
        try:
            # 生成描述的向量表示
            vector = generate_embedding_768(description)
            if not vector:
                return None
            
            # 使用向量相似度搜索
            scenes = self.graph.find_similar_descriptions(
                query_vector=vector,
                label="Scene",
                limit=1,
                min_similarity=0.6  # 设置较低的阈值以提高召回率
            )
            
            if scenes:
                return scenes[0]
            return None
            
        except Exception as e:
            print(f"⚠️ 查找目的地场景失败: {e}")
            return None
    
    def extract_movement_details(
        self,
        command: str
    ) -> Optional[Tuple[str, float]]:
        """从指令中提取移动方向和相对距离，如果没有提取到就根据用户的描述生成"""
        try:
            prompt = f"""
你是一个移动指令分析器。请从以下指令中提取或生成移动方向和相对距离。
即使指令中没有明确的方向和距离，也请根据指令的语义生成合理的移动参数。

方向映射：
- 正北/前方 = N
- 东北 = NE
- 正东/右方 = E
- 东南 = SE
- 正南/后方 = S
- 西南 = SW
- 正西/左方 = W
- 西北 = NW

距离说明：
- 一步 = 1.0
- 很近/一点点 = 0.5
- 很远 = 2.0
- 具体数字直接使用

指令：{command}

请按以下格式返回（仅返回一行，不要其他内容）：
方向,距离

例如：
- 向东走一步 -> E,1.0
- 往西北方向移动一点 -> NW,0.5
- 继续前进 -> N,1.0
- 到处逛逛 -> NE,0.5
- 随便走走 -> E,0.5
"""
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,  # 增加一点随机性
                    "max_tokens": 10
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-tbcaipmckbmbxkjpzmwgfutjtlgsnqecdmlvhlelinckvwok"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()
                if "," in result:
                    direction, distance = result.split(",")
                    try:
                        distance = float(distance)
                        if direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                            return direction, min(max(distance, 0.1), 2.0)
                    except ValueError:
                        pass
            return None
            
        except Exception as e:
            print(f"⚠️ 提取移动细节失败: {e}")
            return None
    
    def convert_direction_to_polar(
        self,
        direction: str,
        distance: float
    ) -> Tuple[float, float]:
        """将方向和距离转换为极坐标增量"""
        # 方向到角度的映射
        direction_angles = {
            "N": 90,    # 北
            "NE": 45,   # 东北
            "E": 0,     # 东
            "SE": 315,  # 东南
            "S": 270,   # 南
            "SW": 225,  # 西南
            "W": 180,   # 西
            "NW": 135   # 西北
        }
        
        # 将角度转换为弧度
        theta = math.radians(direction_angles[direction])
        
        # 计算r和θ的增量
        dr = distance * math.cos(theta)  # r在该方向上的增量
        dtheta = distance * math.sin(theta)  # θ在该方向上的增量
        
        return dr, dtheta
    
    def find_nearby_scenes(
        self,
        position: Tuple[float, float],
        radius: float = 2.0
    ) -> List[Dict]:
        """查找附近的场景"""
        return self.graph.find_nearby_scenes(position, radius)
    
    def get_user_position(self, user_id: str) -> Optional[Tuple[float, float]]:
        """获取用户当前位置"""
        return self.graph.get_user_position(user_id)
    
    def update_user_position(
        self,
        user_id: str,
        position: Tuple[float, float]
    ) -> bool:
        """更新用户位置"""
        return self.graph.update_user_position(user_id, position)
    
    def process_movement(
        self,
        user_id: str,
        command: str
    ) -> Dict:
        """处理移动指令
        
        返回:
            {
                "success": bool,
                "message": str,
                "new_position": Optional[Tuple[float, float]],
                "nearby_scenes": Optional[List[Dict]]
            }
        """
        # 获取用户当前位置
        current_position = self.get_user_position(user_id)
        if not current_position:
            return {
                "success": False,
                "message": "⚠️ 无法获取用户当前位置",
                "new_position": None,
                "nearby_scenes": None
            }
        
        # 尝试提取目的地
        destination = self.extract_destination(command)
        if destination:
            # 使用模糊搜索查找目的地场景
            scenes = self.graph.find_scene_by_description(destination)
            if scenes:
                # 更新用户位置到目标场景
                scene = scenes[0]  # 取第一个匹配的场景
                new_position = scene["position"]
                if self.update_user_position(user_id, new_position):
                    return {
                        "success": True,
                        "message": f"✅ 已到达: {scene['description']}",
                        "new_position": new_position,
                        "nearby_scenes": [scene]
                    }
            
            return {
                "success": False,
                "message": f"⚠️ 找不到目的地: {destination}",
                "new_position": None,
                "nearby_scenes": None
            }
        
        # 如果没有明确目的地，尝试方向移动
        movement = self.extract_movement_details(command)
        if not movement:
            return {
                "success": False,
                "message": "⚠️ 无法理解移动指令",
                "new_position": None,
                "nearby_scenes": None
            }
        
        # 计算新位置
        direction, distance = movement
        dr, dtheta = self.convert_direction_to_polar(direction, distance)
        new_r = current_position[0] + dr
        new_theta = current_position[1] + dtheta
        new_position = (new_r, new_theta)
        
        # 更新用户位置
        if self.update_user_position(user_id, new_position):
            # 查找附近场景
            nearby = self.find_nearby_scenes(new_position)
            return {
                "success": True,
                "message": f"✅ 已移动到 ({new_r:.2f}, {new_theta:.2f})",
                "new_position": new_position,
                "nearby_scenes": nearby
            }
        
        return {
            "success": False,
            "message": "⚠️ 更新位置失败",
            "new_position": None,
            "nearby_scenes": None
        }

# 测试用例
if __name__ == "__main__":
    # 初始化数据库连接
    graph = GraphDB(
        "bolt://localhost:7687",
        "neo4j",
        "password"
    )
    
    # 初始化移动服务
    movement = MovementService(
        graph,
        "http://localhost:11434"
    )
    
    try:
        # 测试目的地提取
        command = "去神秘的洞穴"
        destination = movement.extract_destination(command)
        print(f"\n指令: {command}")
        print(f"提取的目的地: {destination}")
        
        if destination:
            # 测试目的地场景查找
            scenes = movement.graph.find_scene_by_description(destination)
            print("\n找到的场景:")
            if scenes:
                scene = scenes[0]
                print(f"- 描述: {scene['description']}")
                print(f"- 位置: {scene['position']}")
            else:
                print("未找到匹配场景")
        
        # 测试方向移动
        command = "向东北方向走一步"
        result = movement.process_movement("test_user", command)
        print(f"\n指令: {command}")
        print(f"结果: {result['message']}")
        if result['nearby_scenes']:
            print("\n附近场景:")
            for scene in result['nearby_scenes']:
                print(f"- {scene['description']} (距离: {scene['distance']:.2f})")
    
    finally:
        graph.close() 