"""Neo4j 数据库服务

提供图数据库的基本操作接口。
"""

from neo4j import GraphDatabase, Driver
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

class GraphDB:
    """Neo4j 数据库操作封装"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化数据库连接
        
        参数:
            uri: Neo4j 数据库地址
            user: 用户名
            password: 密码
        """
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """关闭数据库连接"""
        self._driver.close()
    
    def _run_write(self, query: str, params: Optional[dict] = None):
        """执行写入操作"""
        with self._driver.session() as session:
            return session.execute_write(lambda tx: list(tx.run(query, params or {})))
    
    def _run_read(self, query: str, params: Optional[dict] = None):
        """执行读取操作"""
        with self._driver.session() as session:
            return session.execute_read(lambda tx: list(tx.run(query, params or {})))
    
    # ---- 用户相关操作 ----
    
    def get_user_position(self, user_id: str) -> Optional[Tuple[float, float]]:
        """获取用户当前位置"""
        query = """
        MATCH (u:User {id: $user_id})
        RETURN u.r as r, u.theta as theta
        """
        
        try:
            result = self._run_read(query, {"user_id": user_id})
            if result and len(result) > 0:
                record = result[0]
                # 确保 r 和 theta 都存在且不为 None
                if record.get("r") is not None and record.get("theta") is not None:
                    return (float(record["r"]), float(record["theta"]))
            
            # 如果找不到用户或位置信息不完整，创建一个新用户在原点
            self.create_user(user_id, (0.0, 0.0))
            return (0.0, 0.0)
            
        except Exception as e:
            print(f"⚠️ 获取用户位置失败: {e}")
            # 出错时也创建一个新用户在原点
            try:
                self.create_user(user_id, (0.0, 0.0))
                return (0.0, 0.0)
            except:
                return None
    
    def update_user_position(self, user_id: str, position: Tuple[float, float]) -> bool:
        """更新用户位置"""
        query = """
        MATCH (u:User {id: $user_id})
        SET u.r = $r, u.theta = $theta
        """
        
        try:
            self._run_write(
                query,
                {
                    "user_id": user_id,
                    "r": position[0],
                    "theta": position[1]
                }
            )
            return True
        except Exception as e:
            print(f"⚠️ 更新用户位置失败: {e}")
            return False
    
    def create_user(self, user_id: str, position: Tuple[float, float]) -> bool:
        """创建新用户，如果已存在则更新位置"""
        query = """
        MERGE (u:User {id: $user_id})
        SET u.r = $r,
            u.theta = $theta
        """
        
        try:
            self._run_write(
                query,
                {
                    "user_id": user_id,
                    "r": position[0],
                    "theta": position[1]
                }
            )
            return True
        except Exception as e:
            print(f"⚠️ 创建/更新用户失败: {e}")
            return False
    
    # ---- 场景相关操作 ----
    
    def find_scene_by_description(
        self,
        description: str,
        limit: int = 5
    ) -> List[Dict]:
        """使用模糊匹配查找场景"""
        query = """
        MATCH (s:Scene)
        WHERE toLower(s.description) CONTAINS toLower($desc)
        RETURN s
        LIMIT $limit
        """
        
        result = self._run_read(
            query,
            {
                "desc": description,
                "limit": limit
            }
        )
        
        return [
            {
                "id": record["s"]["id"],
                "description": record["s"]["description"],
                "position": (record["s"]["r"], record["s"]["theta"])
            }
            for record in result
        ]
    
    def find_nearby_scenes(
        self,
        position: Tuple[float, float],
        radius: float = 2.0
    ) -> List[Dict]:
        """查找给定位置附近的场景"""
        query = """
        MATCH (s:Scene)
        WHERE abs(s.r - $r) <= $radius AND abs(s.theta - $theta) <= $radius
        WITH s, sqrt(power(s.r - $r, 2) + power(s.theta - $theta, 2)) as distance
        ORDER BY distance
        RETURN s, distance
        """
        
        result = self._run_read(
            query,
            {
                "r": position[0],
                "theta": position[1],
                "radius": radius
            }
        )
        
        return [
            {
                "id": record["s"]["id"],
                "description": record["s"]["description"],
                "position": (record["s"]["r"], record["s"]["theta"]),
                "distance": record["distance"]
            }
            for record in result
        ]
    
    def create_scene(
        self,
        scene_id: str,
        description: str,
        position: Tuple[float, float]
    ) -> bool:
        """创建新场景"""
        query = """
        CREATE (s:Scene {
            id: $scene_id,
            description: $description,
            r: $r,
            theta: $theta
        })
        """
        
        try:
            self._run_write(
                query,
                {
                    "scene_id": scene_id,
                    "description": description,
                    "r": position[0],
                    "theta": position[1]
                }
            )
            return True
        except Exception as e:
            print(f"⚠️ 创建场景失败: {e}")
            return False

    # ---- 向量相关操作 ----

    def create_vector_index(self, label: str, property_name: str, dimension: int = 768) -> bool:
        """创建向量索引"""
        query = f"""
        CALL db.index.vector.createNodeIndex(
            $index_name,
            $label,
            $property,
            $dimension,
            'cosine'
        )
        """
        
        try:
            self._run_write(
                query,
                {
                    "index_name": f"{label}_{property_name}_vector_idx",
                    "label": label,
                    "property": property_name,
                    "dimension": dimension
                }
            )
            return True
        except Exception as e:
            print(f"⚠️ 创建向量索引失败: {e}")
            return False
    
    def store_relation_vector(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        vector: List[float],
        properties: Optional[Dict] = None
    ) -> bool:
        """存储关系向量"""
        query = """
        MATCH (from {id: $from_id})
        MATCH (to {id: $to_id})
        CREATE (from)-[r:$relation_type {
            vector: $vector
        }]->(to)
        SET r += $properties
        """
        
        try:
            self._run_write(
                query,
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "relation_type": relation_type,
                    "vector": vector,
                    "properties": properties or {}
                }
            )
            return True
        except Exception as e:
            print(f"⚠️ 存储关系向量失败: {e}")
            return False
    
    def store_description_vector(
        self,
        node_id: str,
        vector: List[float],
        label: Optional[str] = None
    ) -> bool:
        """存储描述向量"""
        query = """
        MATCH (n {id: $node_id})
        SET n.description_vector = $vector
        """
        if label:
            query = f"""
            MATCH (n:{label} {{id: $node_id}})
            SET n.description_vector = $vector
            """
            
        try:
            self._run_write(
                query,
                {
                    "node_id": node_id,
                    "vector": vector
                }
            )
            return True
        except Exception as e:
            print(f"⚠️ 存储描述向量失败: {e}")
            return False
    
    def find_similar_relations(
        self,
        query_vector: List[float],
        relation_type: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """查找相似关系"""
        query = """
        MATCH (from)-[r:$relation_type]->(to)
        WHERE r.vector IS NOT NULL
        WITH from, r, to,
             gds.similarity.cosine(r.vector, $query_vector) AS similarity
        WHERE similarity >= $min_similarity
        RETURN from.id as from_id, to.id as to_id,
               from.description as from_desc, to.description as to_desc,
               type(r) as relation_type, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        result = self._run_read(
            query,
            {
                "relation_type": relation_type,
                "query_vector": query_vector,
                "min_similarity": min_similarity,
                "limit": limit
            }
        )
        
        return [
            {
                "from_id": record["from_id"],
                "to_id": record["to_id"],
                "from_description": record["from_desc"],
                "to_description": record["to_desc"],
                "relation_type": record["relation_type"],
                "similarity": record["similarity"]
            }
            for record in result
        ]
    
    def find_similar_descriptions(
        self,
        query_vector: List[float],
        label: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """查找相似描述"""
        query = """
        MATCH (n)
        WHERE n.description_vector IS NOT NULL
        """
        
        if label:
            query = f"""
            MATCH (n:{label})
            WHERE n.description_vector IS NOT NULL
            """
            
        query += """
        WITH n,
             gds.similarity.cosine(n.description_vector, $query_vector) AS similarity
        WHERE similarity >= $min_similarity
        RETURN n.id as id, n.description as description,
               labels(n) as labels, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        result = self._run_read(
            query,
            {
                "query_vector": query_vector,
                "min_similarity": min_similarity,
                "limit": limit
            }
        )
        
        return [
            {
                "id": record["id"],
                "description": record["description"],
                "labels": record["labels"],
                "similarity": record["similarity"]
            }
            for record in result
        ]
    
    def batch_store_vectors(
        self,
        items: List[Dict[str, Union[str, List[float], Dict]]],
        vector_type: str = "description"
    ) -> bool:
        """批量存储向量
        
        items 格式:
        - description向量: [{"id": str, "vector": List[float], "label": str}]
        - relation向量: [{"from_id": str, "to_id": str, "type": str, 
                        "vector": List[float], "properties": Dict}]
        """
        if vector_type == "description":
            for item in items:
                success = self.store_description_vector(
                    item["id"],
                    item["vector"],
                    item.get("label")
                )
                if not success:
                    return False
        else:  # relation
            for item in items:
                success = self.store_relation_vector(
                    item["from_id"],
                    item["to_id"],
                    item["type"],
                    item["vector"],
                    item.get("properties")
                )
                if not success:
                    return False
        return True

# 测试用例
if __name__ == "__main__":
    # 初始化数据库连接
    graph = GraphDB(
        "bolt://localhost:7687",
        "neo4j",
        "password"
    )
    
    try:
        # 测试创建用户
        test_user_id = "test_user"
        test_position = (0.0, 0.0)
        success = graph.create_user(test_user_id, test_position)
        print(f"\n创建用户: {'成功' if success else '失败'}")
        
        # 测试创建场景
        test_scene_id = "scene_001"
        test_scene_desc = "一个神秘的洞穴入口，周围长满了荧光蘑菇。"
        test_scene_pos = (1.0, math.pi/4)
        success = graph.create_scene(test_scene_id, test_scene_desc, test_scene_pos)
        print(f"\n创建场景: {'成功' if success else '失败'}")
        
        # 测试查找场景
        scenes = graph.find_scene_by_description("洞穴")
        print("\n查找场景结果:")
        for scene in scenes:
            print(f"- {scene['description']}")
        
        # 测试查找附近场景
        nearby = graph.find_nearby_scenes((0.0, 0.0), 2.0)
        print("\n附近场景:")
        for scene in nearby:
            print(f"- {scene['description']} (距离: {scene['distance']:.2f})")
        
    finally:
        graph.close() 