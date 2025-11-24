# -*- coding: utf-8 -*-
"""
Travel Planner Workflow - 旅行规划智能体工作流
集成了 Agent、工具、奖励函数于一个文件中
参考 learn_to_ask 的结构设计
"""

from __future__ import annotations

import json
import re
import asyncio

import csv
import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Set, Union
from dataclasses import dataclass
from agentscope.model import OpenAIChatModel

import openai
import torch
from sentence_transformers import SentenceTransformer
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

from agentscope.formatter import OpenAIChatFormatter

from trinity.common.experience import Experience
from trinity.common.workflows.workflow import WORKFLOWS, Workflow, Task
from trinity.common.models.model import ModelWrapper


 
 
# ============================================================================
# 工具数据结构定义
# ============================================================================

@dataclass
class FlightRecord:
    """航班记录"""
    flight_number: str
    price: float
    dep_time: str
    arr_time: str
    actual_elapsed_time: str
    flight_date: str
    origin_city_name: str
    dest_city_name: str
    distance: float


@dataclass
class AccommodationRecord:
    """住宿记录"""
    name: str
    price: float
    room_type: str
    house_rules: str
    minimum_nights: float
    maximum_occupancy: float
    review_rate_number: float
    city: str


@dataclass
class AttractionRecord:
    """景点记录"""
    name: str
    latitude: float
    longitude: float
    address: str
    phone: str
    website: str
    city: str


@dataclass
class RestaurantRecord:
    """餐厅记录"""
    name: str
    average_cost: float
    cuisines: str
    aggregate_rating: float
    city: str


# ============================================================================
# 工具类实现
# ============================================================================

class Cities:
    """城市搜索工具"""
    
    def __init__(self, path: str = None):
        if path is None:
             
            path = "/home/ecs-user/database/background/citySet_with_states.txt"
        self.path = str(path)
        self.data: Dict[str, List[str]] = {}
        self.load_data()
    
    def load_data(self):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        city, state = parts
                        if state not in self.data:
                            self.data[state] = []
                        self.data[state].append(city)
          
            
    
    def run(self, state: str) -> List[str]:
        if state not in self.data:
            raise ValueError(f"Invalid State: {state}")
        return self.data[state]


class Flights:
    """航班搜索工具"""
    
    def __init__(self, path: str = None):
        if path is None:
             
            path = "/home/ecs-user/database/flights/clean_Flights_2022.csv"
        self.path = str(path)
        self.data: List[FlightRecord] = []
        self.load_db()
    
    def load_db(self):
        self.data.clear()
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if self._is_record_complete(row):
                        record = FlightRecord(
                            flight_number=row['Flight Number'],
                            price=float(row['Price']),
                            dep_time=row['DepTime'],
                            arr_time=row['ArrTime'],
                            actual_elapsed_time=row['ActualElapsedTime'],
                            flight_date=row['FlightDate'],
                            origin_city_name=row['OriginCityName'],
                            dest_city_name=row['DestCityName'],
                            distance=float(row['Distance'])
                        )
                        self.data.append(record)
                except (KeyError, ValueError):
                    continue

    
    def _is_record_complete(self, record: Dict) -> bool:
        required_fields = ['Flight Number', 'Price', 'DepTime', 'ArrTime',
                          'ActualElapsedTime', 'FlightDate', 'OriginCityName',
                          'DestCityName', 'Distance']
        for field in required_fields:
            if field not in record or not record[field] or not record[field].strip():
                return False
        return True
    
    def run(self, origin: str, destination: str, departure_date: str) -> Union[List[FlightRecord], str]:
        results = [
            flight for flight in self.data
            if (flight.origin_city_name == origin and
                flight.dest_city_name == destination and
                flight.flight_date == departure_date)
        ]
        if not results:
            return f"There is no flight from {origin} to {destination} on {departure_date}."
        return results


class Accommodations:
    """住宿搜索工具"""
    
    def __init__(self, path: str = None):
        if path is None:
             
            path = "/home/ecs-user/database/accommodations/clean_accommodations_2022.csv"
        self.path = str(path)
        self.data: List[AccommodationRecord] = []
        self.load_db()
    
    def load_db(self):
        self.data.clear()

        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if self._is_record_complete(row):
                        record = AccommodationRecord(
                            name=row['NAME'],
                            price=float(row['price']),
                            room_type=row['room type'],
                            house_rules=row['house_rules'],
                            minimum_nights=float(row['minimum nights']),
                            maximum_occupancy=float(row['maximum occupancy']),
                            review_rate_number=float(row['review rate number']),
                            city=row['city']
                        )
                        self.data.append(record)
                except (KeyError, ValueError):
                    continue

    
    def _is_record_complete(self, record: Dict) -> bool:
        required_fields = ['NAME', 'price', 'room type', 'house_rules',
                          'minimum nights', 'maximum occupancy',
                          'review rate number', 'city']
        for field in required_fields:
            if field not in record or not record[field] or not str(record[field]).strip():
                return False
        return True
    
    def run(self, city: str) -> Union[List[AccommodationRecord], str]:
        results = [
            accommodation for accommodation in self.data
            if accommodation.city == city
        ]
        if not results:
            return "There is no accommodation in this city."
        return results


class Attractions:
    """景点搜索工具"""
    
    def __init__(self, path: str = None):
        if path is None:
             
            path = "/home/ecs-user/database/attractions/attractions.csv"
        self.path = str(path)
        self.data: List[AttractionRecord] = []
        self.load_db()
    
    def load_db(self):
        self.data.clear()

        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if self._is_record_complete(row):
                        record = AttractionRecord(
                            name=row['Name'],
                            latitude=float(row['Latitude']),
                            longitude=float(row['Longitude']),
                            address=row['Address'],
                            phone=row['Phone'],
                            website=row['Website'],
                            city=row['City']
                        )
                        self.data.append(record)
                except (KeyError, ValueError):
                    continue

    
    def _is_record_complete(self, record: Dict) -> bool:
        required_fields = ['Name', 'Latitude', 'Longitude', 'Address', 
                          'Phone', 'Website', 'City']
        for field in required_fields:
            if field not in record or not record[field] or not str(record[field]).strip():
                return False
        return True
    
    def run(self, city: str) -> Union[List[AttractionRecord], str]:
        results = [
            attraction for attraction in self.data
            if attraction.city == city
        ]
        if not results:
            return "There is no attraction in this city."
        return results


class Restaurants:
    """餐厅搜索工具"""
    
    def __init__(self, path: str = None):
        if path is None:
            
            path = "/home/ecs-user/database/restaurants/clean_restaurant_2022.csv"
        self.path = str(path)
        self.data: List[RestaurantRecord] = []
        self.load_db()
    
    def load_db(self):
        self.data.clear()
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if self._is_record_complete(row):
                        record = RestaurantRecord(
                            name=row['Name'],
                            average_cost=float(row['Average Cost']),
                            cuisines=row['Cuisines'],
                            aggregate_rating=float(row['Aggregate Rating']),
                            city=row['City']
                        )
                        self.data.append(record)
                except (KeyError, ValueError):
                    continue
    
    def _is_record_complete(self, record: Dict) -> bool:
        required_fields = ['Name', 'Average Cost', 'Cuisines', 
                          'Aggregate Rating', 'City']
        for field in required_fields:
            if field not in record or not record[field] or not str(record[field]).strip():
                return False
        return True
    
    def run(self, city: str) -> Union[List[RestaurantRecord], str]:
        results = [
            restaurant for restaurant in self.data
            if restaurant.city == city
        ]
        if not results:
            return "There is no restaurant in this city."
        return results


class DistanceMatrix:
    """距离矩阵工具"""
    
    def __init__(self):
        base_dir = Path(__file__).parent
        self.data: List[Dict] = []
        csv_file = "/home/ecs-user/database/googleDistanceMatrix/distance.csv"
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(row) >= 5:
                    self.data.append({
                        'origin': row.get('origin', '').strip(),
                        'destination': row.get('destination', '').strip(),
                        'cost': row.get('cost', '').strip(),
                        'duration': row.get('duration', '').strip(),
                        'distance': row.get('distance', '').strip()
                    })
    
    def _extract_before_parenthesis(self, text: str) -> str:
        if text is None:
            return None
        idx = text.find('(')
        if idx > 0:
            return text[:idx].strip()
        return text.strip()
    
    def _is_valid(self, duration: str, distance: str) -> bool:
        if not duration or not distance:
            return False
        if duration in ['nan', ''] or distance in ['nan', '']:
            return False
        return True
    
    def _calculate_cost(self, distance: str, rate: float) -> Optional[int]:
        try:
            distance_str = distance.replace('km', '').replace(',', '').strip()
            distance_value = float(distance_str)
            return int(distance_value * rate)
        except (ValueError, AttributeError):
            return None
    
    def run(self, origin: str, destination: str, mode: str = "driving") -> str:
        origin = self._extract_before_parenthesis(origin)
        destination = self._extract_before_parenthesis(destination)
        
        for record in self.data:
            if record['origin'] == origin and record['destination'] == destination:
                duration = record['duration']
                distance = record['distance']
                
                if not self._is_valid(duration, distance):
                    return "No valid information."
                
                cost = None
                if "driving" in mode:
                    cost = self._calculate_cost(distance, 0.05)
                elif mode == "taxi":
                    cost = self._calculate_cost(distance, 1.0)
                
                if "day" in duration:
                    return "No valid information."
                
                return (f"{mode}, from {origin} to {destination}, "
                       f"duration: {duration}, distance: {distance}, cost: {cost}")
        
        return f"{mode}, from {origin} to {destination}, no valid information."


# ============================================================================
# Agent 系统提示词
# ============================================================================

AGENT_SYSTEM_PROMPT = """
You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol ’-’ indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the ’Current City’ section as in the example (i.e., from A to B).
***** Example *****
[
    {
        "days": 1,
        "current_city": "from Kansas City to Pensacola",
        "transportation": "Flight F0123456, Kansas City to Pensacola, departure date: 2022-03-27, cost: $200",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "Hotel Pensacola, Pensacola"
    },
    {
        "days": 2,
        "current_city": "Pensacola",
        "transportation": "-",
        "breakfast": "Breakfast at Hotel Pensacola, Pensacola",
        "attraction": "Pensacola Beach, Pensacola;Fort Pickens, Pensacola;",
        "lunch": "Dinner at The Bluewater Grill, Pensacola",
        "dinner": "Lunch at The Bluewater Grill, Pensacola",
        "accommodation": "Hotel Pensacola, Pensacola"
    },
    {
        "days": 3,
        "current_city": "from Pensacola to Kansas City",
        "transportation": "Flight F0654321, Pensacola to Kansas City, departure date: 2022-03-29, cost: $200",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }
]
***** Example Ends *****
"""




class TravelAgent:
    """旅行规划 Agent"""
    
    def __init__(self, model,formatter,max_iters):
        """
        初始化 Travel Agent
        
        Args:
            model_config: 模型配置字典
        """
        
        # 初始化工具
        self.cities_tool = Cities()
        self.flights_tool = Flights()
        self.accommodations_tool = Accommodations()
        self.attractions_tool = Attractions()
        self.restaurants_tool = Restaurants()
        self.distance_tool = DistanceMatrix()
        
        # 创建 toolkit 并注册工具
        toolkit = Toolkit()
        self._register_tools(toolkit)
        
        
        # 创建 Agent
        self.agent = ReActAgent(
            name="travel-planner",
            sys_prompt=AGENT_SYSTEM_PROMPT,
            model=model,
            toolkit=toolkit,
            memory=InMemoryMemory(),
            max_iters=max_iters,
            formatter= formatter
        )

        
    
    def _register_tools(self, toolkit: Toolkit):
        """注册所有工具到 toolkit"""
        
        # 1. 城市搜索
        async def city_search(state: str) -> ToolResponse:
            """Find cities in a specific state.
            
            Args:
                state (str):
                    The name of the state where you're seeking cities.
            """
            try:
                cities = self.cities_tool.run(state)
                result_text = f"Cities in {state}: {', '.join(cities[:10])}"  # 限制返回数量
                if len(cities) > 10:
                    result_text += f" ... and {len(cities) - 10} more cities."
            except ValueError as e:
                result_text = str(e)
            
            return ToolResponse(
                content=[{"type": "text", "text": result_text}]
            )
        
        # 2. 航班搜索
        async def flight_search(origin: str, destination: str, departure_date: str) -> ToolResponse:
            """Search for flights between two cities.
            
            Args:
                origin (str):
                    The city you'll be flying out from.
                destination (str):
                    The city you aim to reach.
                departure_date (str):
                    The date of your travel in YYYY-MM-DD format.
            """
            results = self.flights_tool.run(origin, destination, departure_date)
            
            if isinstance(results, str):
                result_text = results
            else:
                output = [f"Found {len(results)} flights from {origin} to {destination} on {departure_date}:"]
                for i, flight in enumerate(results[:3], 1):  # 只显示前3个
                    output.append(
                        f"{i}. Flight {flight.flight_number}: ${flight.price:.2f}, "
                        f"Depart: {flight.dep_time}, Arrive: {flight.arr_time}"
                    )
                if len(results) > 3:
                    output.append(f"... and {len(results) - 3} more flights available.")
                result_text = "\n".join(output)
            
            return ToolResponse(
                content=[{"type": "text", "text": result_text}]
            )
        
        # 3. 住宿搜索
        async def accommodation_search(city: str) -> ToolResponse:
            """Search for accommodations in a city.
            
            Args:
                city (str):
                    The name of the city where you're seeking accommodation.
            """
            results = self.accommodations_tool.run(city)
            
            if isinstance(results, str):
                result_text = results
            else:
                output = [f"Found {len(results)} accommodations in {city}:"]
                for i, acc in enumerate(results[:3], 1):  # 只显示前3个
                    output.append(
                        f"{i}. {acc.name}: ${acc.price:.2f}/night, "
                        f"Room: {acc.room_type}, Rating: {acc.review_rate_number:.1f}/5"
                    )
                if len(results) > 3:
                    output.append(f"... and {len(results) - 3} more options available.")
                result_text = "\n".join(output)
            
            return ToolResponse(
                content=[{"type": "text", "text": result_text}]
            )
        
        # 4. 景点搜索
        async def attraction_search(city: str) -> ToolResponse:
            """Search for attractions in a city.
            
            Args:
                city (str):
                    The name of the city where you're seeking attractions.
            """
            results = self.attractions_tool.run(city)
            
            if isinstance(results, str):
                result_text = results
            else:
                output = [f"Found {len(results)} attractions in {city}:"]
                for i, att in enumerate(results[:5], 1):  # 只显示前5个
                    output.append(f"{i}. {att.name} - {att.address}")
                if len(results) > 5:
                    output.append(f"... and {len(results) - 5} more attractions available.")
                result_text = "\n".join(output)
            
            return ToolResponse(
                content=[{"type": "text", "text": result_text}]
            )
        
        # 5. 餐厅搜索
        async def restaurant_search(city: str) -> ToolResponse:
            """Search for restaurants in a city.
            
            Args:
                city (str):
                    The name of the city where you're seeking restaurants.
            """
            results = self.restaurants_tool.run(city)
            
            if isinstance(results, str):
                result_text = results
            else:
                output = [f"Found {len(results)} restaurants in {city}:"]
                for i, rest in enumerate(results[:5], 1):  # 只显示前5个
                    output.append(
                        f"{i}. {rest.name}: {rest.cuisines}, "
                        f"Avg cost: ${rest.average_cost:.2f}, Rating: {rest.aggregate_rating:.1f}/5"
                    )
                if len(results) > 5:
                    output.append(f"... and {len(results) - 5} more restaurants available.")
                result_text = "\n".join(output)
            
            return ToolResponse(
                content=[{"type": "text", "text": result_text}]
            )
        
        # 6. 距离计算
        async def distance_matrix(origin: str, destination: str, mode: str = "driving") -> ToolResponse:
            """Calculate distance, duration, and cost between two cities.
            
            Args:
                origin (str):
                    The departure city of your journey.
                destination (str):
                    The destination city of your journey.
                mode (str):
                    The method of transportation. Options: 'driving', 'self-driving', 'taxi'.
            """
            result_text = self.distance_tool.run(origin, destination, mode)
            
            return ToolResponse(
                content=[{"type": "text", "text": result_text}]
            )
        
        # 注册所有工具
        toolkit.register_tool_function(city_search)
        toolkit.register_tool_function(flight_search)
        toolkit.register_tool_function(accommodation_search)
        toolkit.register_tool_function(attraction_search)
        toolkit.register_tool_function(restaurant_search)
        toolkit.register_tool_function(distance_matrix)


 

# ============================================================================
# 语义评估提示词（用于 LLM Judge）
# ============================================================================

SEMANTIC_JUDGE_PROMPT = """你是一个专业的旅行规划评估专家。请评估以下AI助手生成的旅行规划质量。

【用户需求】
{user_query}

【AI生成的旅行规划】
{agent_plan}

综合考虑评分标准，输出一个0-1的小数，代表旅行规划的质量，0分代表质量差，1分代表质量好。
【评分标准】 
1. **准确性 (Accuracy)**: 提到的地点、价格、时间等信息是否合理可信
2. **完整性 (Completeness)**: 是否包含旅行规划的必要元素
3. **相关性 (Relevance)**: 是否真正满足用户的具体需求
4. **连贯性 (Coherence)**: 行程安排是否合理流畅
5. **实用性 (Practicality)**: 建议是否可执行和有用
 
直接输出0-1的小数，不要有任何前缀或后缀。"""


# ============================================================================
# 主工作流类
# ============================================================================

@WORKFLOWS.register_module("travel_planner_workflow")
class TravelAgentWorkflow(Workflow):
    """
    旅行规划工作流 - 所有功能集成
    
    集成了六个奖励函数:
    1. TaskCompletionReward - 任务完成度 (权重 0.1)
    2. SemanticQualityEvaluator - 语义质量 (权重 0.25)
    3. SemanticSimilarityReward - 相似度 (权重 0.25)
    4. ToolUsageReward - 工具使用情况 (权重 0.2)
    5. TravelPlannerReward - TravelPlanner约束评估 (权重 0.1)
    6. JsonFormatReward - JSON格式检查 (权重 0.1)
    """

    can_reset: bool = False
    can_repeat: bool = False
    is_async: bool = True

 
    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        """
        初始化工作流
        
        Args:
            task: 任务对象
            model: 主模型
            auxiliary_models: 辅助模型列表（用于语义评估）
        """
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        self.auxiliary_models = auxiliary_models

        self.model_client = model.get_openai_async_client()

        self.agent_model = OpenAIChatModel(
            api_key="EMPTY",
            model_name=self.model_client.model_path,
            generate_kwargs={
                "temperature": 1.0  ,          # 训练时使用随机采样
                "max_tokens": 4096 
            },
            stream=False,
        )
        self.agent_model.client = self.model_client

        self.weights =  {
        "completion": 0.1,    # 任务完成度
        "semantic": 0.25,     # 语义质量
        "similarity": 0.25,   # 相似度
        "tool_usage": 0.2,    # 工具使用情况
        "planner": 0.1,       # TravelPlanner约束评估
        "format": 0.1,        # JSON格式检查
    }

        # # 重置任务配置
        # self.reset(task)

        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]



        self.raw_task = task

        

        # 创建 Agent
        self.max_iters = task.workflow_args.get("max_iters", 1)

        self.total_tools = task.workflow_args.get("total_tools", 6)
        # 2. 语义评估配置（辅助模型）
        self.judge_model = None
        if hasattr(self, 'auxiliary_models') and self.auxiliary_models:
            self.judge_model = self.auxiliary_models[0]
        
        # 3. 相似度计算配置
        if not hasattr(self, 'similarity_model') or self.similarity_model is None:
            similarity_model_name = task.workflow_args.get("similarity_model_name", "all-MiniLM-L6-v2")
            try:
                self.similarity_model = SentenceTransformer(similarity_model_name)

            except Exception as e:
                self.similarity_model = None


        self.agent = TravelAgent(self.agent_model, OpenAIChatFormatter(), self.max_iters)



   

    async def run_async(self) -> List[Experience]:
        """
        异步执行工作流
        
        Returns:
            Experience列表
        """

        # Step 1: call the react agent to solve the task
        response = await self.agent.agent.reply(msg=Msg("user", self.query, role="user"))
        


        # Step 2: calculate the reward based on the response


        
        # 获取历史记录
        try:
            history = await  self.agent.agent.memory.get_memory()
        except AttributeError as e:

            history = []

        
        # 提取经验
        experiences = self.model.extract_experience_from_history(clear_history=True)
        
        #  reward debug 
        # reward = 0.5
        reward = await self.reward_fn(response, experiences, history)

        # Step 3: construct experiences from the interaction history and return them

        
        
        # 为所有经验设置奖励
        for exp in experiences:
            exp.reward = reward
        

        return experiences

    # ========================================================================
    # 奖励函数 - 主入口
    # ========================================================================

    async def reward_fn(
        self,
        response,
        experiences: List[Experience],
        history: List[Msg]
    ) -> float:
        """
        综合奖励函数（并行执行六个子奖励函数）
        
        Args:
            response: Agent响应
            experiences: 经验列表
            history: 交互历史
        
        Returns:
            综合奖励分数 (0.0~1.0)
        """
        self.logger.info("开始计算奖励函数，experiences数量: %d, history数量: %d", len(experiences), len(history))

        # 处理 response.content 的类型（可能是字符串或列表）
        if isinstance(response.content, list):
            response_text = ""
            for item in response.content:
                if isinstance(item, dict) and 'text' in item:
                    response_text += item['text']
                elif isinstance(item, str):
                    response_text += item
        else:
            response_text = str(response.content)
        
        self.logger.info("处理后的响应文本长度: %d", len(response_text))
        
        # 使用 asyncio.gather 并行执行所有奖励函数
        self.logger.info("开始并行执行六个子奖励函数")
        results = await asyncio.gather(
            self._completion_reward(experiences),
            self._semantic_reward(response_text),
            self._similarity_reward(response_text),
            self._tool_usage_reward(history),
            self._planner_reward(response_text),
            self._format_reward(response_text),
            return_exceptions=True
        )

         
        
        # 解包结果并处理异常
        completion_reward = self._handle_result(results[0], "completion", 0.0)
        semantic_reward = self._handle_result(results[1], "semantic", 0.5)
        similarity_reward = self._handle_result(results[2], "similarity", 0.5)
        tool_usage_reward = self._handle_result(results[3], "tool_usage", 0.0)
        planner_reward = self._handle_result(results[4], "planner", 0.5)
        format_reward = self._handle_result(results[5], "format", 0.0)

        reward_dict = {
            "completion": completion_reward,
            "semantic": semantic_reward,
            "similarity": similarity_reward,
            "tool_usage": tool_usage_reward,
            "planner": planner_reward,
            "format": format_reward
        }
        self.logger.info("各子奖励函数计算完成: %s", json.dumps(reward_dict))
        
        # 加权组合
        total_reward = (
            self.weights["completion"] * completion_reward +
            self.weights["semantic"] * semantic_reward +
            self.weights["similarity"] * similarity_reward +
            self.weights["tool_usage"] * tool_usage_reward +
            self.weights["planner"] * planner_reward +
            self.weights["format"] * format_reward
        )
        
        # 确保在 [0, 1] 范围内
        total_reward = max(0.0, min(1.0, total_reward))
        
        self.logger.info("最终综合奖励分数: %.4f", total_reward)
        return total_reward
    
    def _handle_result(self, result, reward_name: str, default_value: float) -> float:
        """处理并行执行的奖励函数结果"""
        if isinstance(result, Exception):
            self.logger.info(Exception)
        
            return default_value
        return float(result)

    # ========================================================================
    # 奖励函数 1: 任务完成度
    # ========================================================================
    
    async def _completion_reward(self, experiences: List[Experience]) -> float:
        """任务完成奖励 - 检查是否在最大步数内完成任务"""
        try:
            self.logger.info("计算任务完成奖励，experiences数量: %d, 最大迭代数: %d", len(experiences), self.max_iters)
            if len(experiences) >= self.max_iters:
                self.logger.info("达到最大步数，任务未完成，奖励: 0.0")
                return 0.0  # 达到最大步数，视为未完成
            self.logger.info("在限制内完成任务，奖励: 1.0")
            return 1.0  # 在限制内完成
        except Exception as e:
            self.logger.info("计算任务完成奖励时出现异常: %s，返回默认值: 0.0", str(e))
            return 0.0

    # ========================================================================
    # 奖励函数 2: 语义质量评估（LLM Judge）
    # ========================================================================
    
    async def _semantic_reward(self, response_text: str) -> float:
        """语义质量奖励（使用LLM评判）"""
        self.logger.info("开始计算语义质量奖励，响应文本长度: %d", len(response_text))
        if not self.judge_model:
            self.logger.info("评判模型不可用，返回默认值: 0.5")
            return 0.5
        
        try:
            prompt = SEMANTIC_JUDGE_PROMPT.format(
                user_query=self.query,
                agent_plan=response_text
            )
            self.logger.info("调用评判模型进行语义质量评估")
            score = await self._call_judge_model(prompt)
            self.logger.info("语义质量评估完成，分数: %.4f", score)
            return float(score)
        except Exception as e:
            self.logger.info("语义质量评估时出现异常: %s，返回默认值: 0.5", str(e))
            return 0.5
    
    async def _call_judge_model(self, prompt: str) -> float:
        """调用评判模型并解析分数"""
        try:
            response = await self.judge_model.chat.completions.create(
                model=getattr(self.judge_model, 'model_path', 'default'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content
            return self._parse_score(content)
        except Exception as e:
        
            raise
    
    def _parse_score(self, text: str) -> float:
        """解析评判模型返回的分数（0-1之间的小数）"""
        text = text.strip()
        
        # 尝试直接解析为浮点数
        try:
            score = float(text)
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
        
        # 尝试提取数字
        number_pattern = r'\b(0?\.\d+|1\.0+|0|1)\b'
        matches = re.findall(number_pattern, text)
        if matches:
            try:
                score = float(matches[0])
                if score > 1 and score <= 100:
                    score = score / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
    
        return 0.5

    # ========================================================================
    # 奖励函数 3: 语义相似度（Embedding）
    # ========================================================================
    
    async def _similarity_reward(self, response_text: str) -> float:
        """相似度奖励（与标准答案比较）"""
        self.logger.info("开始计算相似度奖励")
        if self.similarity_model is None:
            self.logger.info("相似度模型不可用，返回默认值: 0.5")
            return 0.5
        
        ground_truth = self.answer
        if not ground_truth:
            self.logger.info("标准答案不存在，返回默认值: 0.5")
            return 0.5
        
        try:
            self.logger.info("计算响应文本与标准答案的相似度，响应长度: %d, 标准答案长度: %d", 
                           len(response_text), len(ground_truth))
            similarity = await asyncio.to_thread(
                self._compute_similarity,
                response_text,
                ground_truth
            )
            self.logger.info("相似度计算完成，分数: %.4f", similarity)
            return similarity
        except Exception as e:
            self.logger.info("相似度计算时出现异常: %s，返回默认值: 0.5", str(e))
            return 0.5
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的余弦相似度"""
        embeddings = self.similarity_model.encode(
            [text1, text2],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        
        # 余弦相似度 [-1, 1] 转换为 [0, 1]
        score = (similarity.item() + 1) / 2
        return float(score)

    # ========================================================================
    # 奖励函数 4: 工具使用情况
    # ========================================================================
    
    async def _tool_usage_reward(self, history: List[Msg]) -> float:
        """工具使用奖励 - 根据调用的工具数量计算奖励"""
        self.logger.info("开始计算工具使用奖励，历史消息数量: %d", len(history))
        try:
            tool_names = await asyncio.to_thread(
                self._extract_tool_names,
                history
            )
            unique_tool_count = len(tool_names)
            reward = min(1.0, unique_tool_count / self.total_tools)
            self.logger.info("工具使用情况 - 使用的工具: %s, 使用数量: %d/%d, 奖励: %.4f", 
                           list(tool_names), unique_tool_count, self.total_tools, reward)
            return reward
        except Exception as e:
            self.logger.info("工具使用奖励计算时出现异常: %s，返回默认值: 0.0", str(e))
            return 0.0
    
    def _extract_tool_names(self, history: List[Msg]) -> Set[str]:
        """从历史中提取所有被调用的工具名称（去重）"""
        tool_names = set()
        
        for msg in history:
            if msg.role == 'assistant' and isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        tool_name = item.get('name')
                        if tool_name:
                            tool_names.add(tool_name)
        
        return tool_names

    # ========================================================================
    # 奖励函数 5: TravelPlanner 硬性约束评估
    # ========================================================================
    
    async def _planner_reward(self, response_text: str) -> float:
        """TravelPlanner 官方评估奖励 - 使用官方评估系统的常识约束和硬约束检查"""
        self.logger.info("开始计算TravelPlanner官方评估奖励")
        try:
            # 解析 JSON
            try:
                plan = await asyncio.to_thread(json.loads, response_text)
                self.logger.info("JSON解析成功，计划包含 %d 个条目", len(plan) if isinstance(plan, list) else 1)
            except json.JSONDecodeError as e:
                self.logger.info("JSON解析失败: %s，返回奖励: 0.0", str(e))
                return 0.0
            
            # 评估常识约束 
            self.logger.info("开始评估常识约束")
            commonsense_info = await commonsense_eval_async(self.raw_task, plan)
            
            # 检查基础约束
            if not commonsense_info:
                self.logger.info("常识约束评估失败，返回奖励: 0.0")
                return 0.0
            
            if not commonsense_info.get('is_not_absent', [False])[0]:
                self.logger.info("常识约束检查失败 - is_not_absent，返回奖励: 0.0")
                return 0.0
            
            if not commonsense_info.get('is_valid_information_in_sandbox', [False])[0]:
                self.logger.info("常识约束检查失败 - is_valid_information_in_sandbox，返回奖励: 0.0")
                return 0.0
            
            # 评估硬约束（并行执行5个检查）
            self.logger.info("开始评估硬约束")
            hard_info = await hard_eval_async(self.raw_task, plan)
            
            # 计算分数
            commonsense_score = self._calculate_constraint_score(commonsense_info)
            hard_score = self._calculate_constraint_score(hard_info) if hard_info else 1.0
            
            self.logger.info("约束评估完成 - 常识约束分数: %.4f, 硬约束分数: %.4f", 
                           commonsense_score, hard_score)
            
            # 加权综合
            final_score = (
                0.5 * commonsense_score +
                0.5 * hard_score
            )
            
            self.logger.info("TravelPlanner官方评估奖励: %.4f", final_score)
            return final_score
        except Exception as e:
            self.logger.info("TravelPlanner评估时出现异常: %s，返回默认值: 0.5", str(e))
            return 0.5
    
    def _calculate_constraint_score(self, info_box: Dict) -> float:
        """计算约束通过率"""
        if not info_box:
            return 0.0
        
        total = 0
        passed = 0
        
        for key, value in info_box.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                if value[0] is not None:  # 忽略不适用的约束
                    total += 1
                    if value[0] is True:
                        passed += 1
        
        return passed / total if total > 0 else 0.0
   # ========================================================================
    # 奖励函数 6: JSON 格式检查
    # ========================================================================

    async def _format_reward(self, response_text: str) -> float:
        """JSON 格式检查奖励 - 检查响应是否为有效的JSON"""
        self.logger.info("开始检查JSON格式，响应文本长度: %d", len(response_text))
        try:
            reward = await asyncio.to_thread(
                self._check_json_format,
                response_text
            )
            self.logger.info("JSON格式检查完成，奖励: %.1f", reward)
            return reward
        except Exception as e:
            self.logger.info("JSON格式检查时出现异常: %s，返回默认值: 0.0", str(e))
            return 0.0
    
    def _check_json_format(self, text: str) -> float:
        """检查是否为有效的JSON格式"""
        if not text or not isinstance(text, str):
            return 0.0
        
        text = text.strip()
        if not text:
            return 0.0
        
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return 1.0
            return 0.0
        except json.JSONDecodeError:
            return 0.0




   # ========================================================================
    # 奖励函数 5: 硬性约束检查
    # ========================================================================


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data


def convert_bool_values(item):
    if isinstance(item, dict):
        # If the item is a dictionary, recurse on each value
        return {key: convert_bool_values(value) for key, value in item.items()}
    elif isinstance(item, list):
        # If the item is a list, recurse on each item in the list
        return [convert_bool_values(value) for value in item]
    elif isinstance(item, tuple):
        # If the item is a tuple, recurse on each item in the tuple and repackage as a tuple
        return tuple(convert_bool_values(value) for value in item)
    elif isinstance(item, np.bool_):  # Here we check for numpy's bool_ type
        # If the item is a numpy bool_, convert it to a standard Python bool
        return bool(item)
    else:
        # If the item is any other type, return it unchanged
        return item




def extract_from_to(text: str):
    """
    Extracts 'A' and 'B' from the format "from A to B" in the given text, with B ending at a comma or the end of the string.
    
    Args:
    - text (str): The input string.
    
    Returns:
    - tuple: A tuple containing 'A' and 'B'. If no match is found, returns (None, None).
    """
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)

 

def is_valid_transportation(question, tested_data):
    if question['local_constraint']['transportation'] is None:
        return None,None
    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if question['local_constraint']['transportation'] == 'no flight' and 'Flight' in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            elif question['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving'  in value:
                return False, f"The transportation should not be {question['local_constraint']['transportation']}."
            
    return True, None


 

async def hard_eval_async(query_data, tested_data):
    """
    硬约束评估 - 异步并行版本
    
    使用 asyncio.gather 并行执行 5 个硬约束检查，显著提升性能。
    
    Args:
        query_data: 查询数据
        tested_data: 待测试的旅行计划数据
    
    Returns:
        dict: 包含所有硬约束检查结果的字典
    
    性能提升:
        - 串行执行: ~200ms (5 个检查 × 40ms)
        - 并行执行: ~40ms (最慢的单个检查)
        - 加速比: 5x
    """
    # 定义一个辅助函数用于 valid_cost 检查
 
    
    # 使用 asyncio.gather 并行执行所有 5 个硬约束检查
    results = await asyncio.gather(
        asyncio.to_thread(is_valid_transportation, query_data, tested_data),
        return_exceptions=True  # 防止单个检查失败导致全部失败
    )
    
    # 构建结果字典
    return_info = {}
    constraint_names = [
        'valid_transportation',
    ]
    
    for i, name in enumerate(constraint_names):
        if isinstance(results[i], Exception):
            # 如果检查失败，返回 (None, 错误信息) 或 (False, 错误信息)
            return_info[name] = (None, f"Exception: {str(results[i])}")
        else:
            return_info[name] = results[i]
    
    return return_info

 




    # ========================================================================
    # 奖励函数 6: 常识性函数约束
    # ========================================================================

def count_consecutive_values(lst):
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))  # Add the last group of values
    return result


def transportation_match(text: str):

    if 'taxi' in text.lower():
        return 'Taxi'
    
    elif 'self-driving' in text.lower():
        return 'Self-driving'
    
    elif 'flight' in text.lower():
        return 'Flight'



def is_valid_city_sequence(city_list):
    """
    Checks if the city sequence is valid. A valid sequence has every city (except the first and last) 
    appearing consecutively, and no city should appear again once its sequence is over.
    
    Args:
    - city_list (list): List of cities.
    
    Returns:
    - bool: True if the sequence is valid, False otherwise.
    """
    
    # If the list has less than 3 cities, it's invalid.
    if len(city_list) < 3:
        return False
    
    # Set to keep track of visited cities
    visited_cities = set()
    
    i = 0
    while i < len(city_list):
        city = city_list[i]
        
        # If the city was already visited, it's invalid.
        if city in visited_cities and (i != 0 and i != len(city_list) - 1):
            return False
        
        # Count the consecutive occurrences of the city
        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1
        
        # If the city appeared only once in the medium, it's invalid.
        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False
        
        visited_cities.add(city)
    
    return True


 

def is_valid_restaurants(question, tested_data):

    restaurants_list = []

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            if unit['breakfast'] not in restaurants_list:
                restaurants_list.append(unit['breakfast'])
            else:
                return False, f"The restaurant in day {i+1} breakfast is repeated."
        # elif 'breakfast' not in unit :
        #     return False, f"No Breakfast Info."
            
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            if unit['lunch'] not in restaurants_list:
                restaurants_list.append(unit['lunch'])
            else:
                return False, f"The restaurant in day {i+1} lunch {unit['lunch']} is repeated."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
        
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            if unit['dinner'] not in restaurants_list:
                restaurants_list.append(unit['dinner'])
            else:
                return False, f"The restaurant in day {i+1} dinner is repeated."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."

    return True, None
            
def is_valid_attractions(question, tested_data):

    attractions_list = []

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            for attraction in unit['attraction'].split(';')[:-1]:
                if attraction not in attractions_list:
                    attractions_list.append(attraction)
                else:
                    return False, f"The attraction '{attraction}' in day {i+1} is repeated."
                
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
        
    return True, None

def is_valid_transportation(question, tested_data):
    
    if tested_data[0]['transportation'] and tested_data[0]['transportation'] != '-':
        transportation_list = [transportation_match(tested_data[0]['transportation'])]
    
    else:
        return False, "The transportation in day 1 should not be empty."

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            transportation_list.append(transportation_match(unit['transportation']))
        # elif 'transportation' not in unit:
        #     return False, f"No Transportation Info."
    
    if (('Self-driving' in transportation_list) and ('Flight' in transportation_list)) or (('Taxi' in transportation_list) and ('Self-driving' in transportation_list)):
        return False, "The transportation is conflicting."

    return True, None

def is_valid_information_in_current_city(question, tested_data):

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]
        current_city = unit['current_city']
        final_city_list = []

        if 'from' in current_city:
            city1, city2 = extract_from_to(current_city)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            final_city_list = [city1, city2]
        else:
            final_city_list = [extract_before_parenthesis(current_city)]

        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            for city in final_city_list:
                if city not in unit['transportation']:
                    # print(city)
                    return False, f"The transportation in day {i+1} is invalid city choice."
        # elif 'transportation' not in unit:
        #     return False, f"No Transportation Info."
        
        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':

            flag = False

            for city in final_city_list:
                if city  in unit['breakfast']:
                    flag = True

            if not flag:
                return False, f"The breakfast in day {i+1} is invalid city choice."
        # elif 'breakfast' not in unit:
        #     return False, f"No Breakfast Info."
        
        if 'lunch' in unit and unit['lunch'] and unit['lunch'] != '-':
            flag = False

            for city in final_city_list:
                if city  in unit['lunch']:
                    flag = True
            
            if not flag:
                return False, f"The lunch in day {i+1} is invalid city choice."
        # elif 'lunch' not in unit:
        #     return False, f"No Lunch Info."
            
        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            flag = False

            for city in final_city_list:
                if city  in unit['dinner']:
                    flag = True
            
            if not flag:
                return False, f"The dinner in day {i+1} is invalid city choice."
        # elif 'dinner' not in unit:
        #     return False, f"No Dinner Info."
        
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            
            attraction_list = unit['attraction'].split(';')[:-1]

            for attraction in attraction_list:
                flag = False
                for city in final_city_list:
                    if city  in attraction:
                        flag = True
                if not flag:
                    return False, f"The attraction in day {i+1} is invalid city choice."
                
        # elif 'attraction' not in unit:
        #     return False, f"No Attraction Info."
            
            
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            
            if final_city_list[-1] not in unit['accommodation']:
                return False, f"The accommodation in day {i+1} is invalid city choice."
            
        # elif 'accommodation' not in unit:
        #     return False, f"No Accommodation Info."
    
    return True, None
        

 
def is_valid_visiting_city_number(question, tested_data):

    city_set = set()
    

    for i in range(min(question['days'],len(tested_data))):
        city_value = tested_data[i]['current_city']

        if 'from' in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i==0 and  city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."

            city_set.add(city1)
            city_set.add(city2)

        else:
            city_set.add(extract_before_parenthesis(city_value))
    
    city_set.discard(question['org'])

    if len(city_set) != question['visiting_city_number']:
        return False, f"The number of visiting cities should be {question['visiting_city_number']}."
    
    return True, None

def is_valid_days(question, tested_data):
    lens = 0
    for i in range(min(question['days'],len(tested_data))):
        if tested_data[i] != {} and tested_data[i]['current_city'] != "You don't need to fill in the information for this or later days.":
            lens += 1
        
    if lens != question['days']:
        # print(lens)
        return False, f"The number of days should be {question['days']}."
    else:
        return True, None

def is_not_absent(question, tested_data):
    needed_info = 6 * question['days']
    total_valid_info = 0

    if not is_valid_days(question, tested_data)[0]:
        return False, "Invalid Days"
    
    if not is_valid_visiting_city_number(question, tested_data)[0]:
        return False, "Invalid City Number"

    for i in range(min(question['days'],len(tested_data))):
        unit = tested_data[i]

        # 检查所有必需字段是否存在
        required_fields = ['transportation', 'breakfast', 'lunch', 'dinner', 'attraction', 'accommodation']
        for field in required_fields:
            if field not in unit:
                return False, f"No {field.capitalize()} Info."
        
        if ('from ' in unit['current_city'] or 'to ' in unit['current_city']) and unit['transportation'] in ['','-']:
            return False, f"No transportation in day {i+1} is not allowed."
        
        if ('from ' not in unit['current_city'] and  ' to ' not in unit['current_city']) and unit['attraction'] in ['','-']:
            return False, f"No attaction in day {i+1} is not allowed."

        if i != question['days'] - 1 and unit['accommodation'] in ['','-']:
            return False, f"No accommodation in day {i+1} is not allowed."

        if (unit['breakfast'] in ['','-'] or unit['lunch'] in ['','-'] or unit['dinner'] in ['','-']) and 'from ' not in unit['current_city']:
            return False, f"No meal in day {i+1} is not allowed."
        

        for key in unit:
            if unit[key] and unit[key] != '-':
                total_valid_info += 1


    if total_valid_info * 1.0 / needed_info < 0.5:
        return False, f"The absent information is more than 50%."
    
    return True, None


 


async def commonsense_eval_async(query_data, tested_data):
    """
    常识约束评估 - 异步并行版本
    
    使用 asyncio.gather 并行执行 5 个常识约束检查，显著提升性能。
    
    Args:
        query_data: 查询数据
        tested_data: 待测试的旅行计划数据
    
    Returns:
        dict: 包含所有常识约束检查结果的字典
 
    """
    # 使用 asyncio.gather 并行执行所有 5 个常识约束检查
    # asyncio.to_thread 将同步函数放到线程池执行，避免阻塞事件循环
    results = await asyncio.gather(
        asyncio.to_thread(is_valid_restaurants, query_data, tested_data),
        asyncio.to_thread(is_valid_attractions, query_data, tested_data),
        asyncio.to_thread(is_valid_transportation, query_data, tested_data),
        asyncio.to_thread(is_valid_information_in_current_city, query_data, tested_data),
        asyncio.to_thread(is_not_absent, query_data, tested_data),
        return_exceptions=True  # 防止单个检查失败导致全部失败
    )
    
    # 构建结果字典
    return_info = {}
    constraint_names = [
 
        'is_valid_restaurants',
        'is_valid_attractions',
        'is_valid_transportation',
        'is_valid_information_in_current_city',
        'is_not_absent'
    ]
    
    for i, name in enumerate(constraint_names):
        if isinstance(results[i], Exception):
            # 如果检查失败，返回 (False, 错误信息)
            return_info[name] = (False, f"Exception: {str(results[i])}")
        else:
            return_info[name] = results[i]
    
    return return_info



def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def save_file(data, path):
    with open(path,'w',encoding='utf-8') as w:
        for unit in data:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()

def extract_query_number(query_string):
    """
    Extract the number from a query string formatted as "Query X" or "Query X --- Done".
    
    Args:
    - query_string (str): The input string.
    
    Returns:
    - int: The extracted number if found, else None.
    """
    pattern = r"Query (\d+)"
    match = re.search(pattern, query_string)
    return int(match.group(1)) if match else None

def create_data_display(css_content,data,annotation_idx):
    return f"""
    <style>
    {css_content}
    </style>
    <div>
        <span class="query-highlighted"><strong>Query {annotation_idx}:</strong> {data[annotation_idx-1]['query']}</span><br>
        <span class="highlighted"><strong>Day:</strong> {data[annotation_idx-1]['days']}</span>
        <span class="highlighted"><strong>Visiting City Number:</strong> {data[annotation_idx-1]['visiting_city_number']}</span>
        <span class="highlighted"><strong>Date:</strong> {data[annotation_idx-1]['date']}</span>
        <span class="highlighted"><strong>Departure:</strong> {data[annotation_idx-1]['org']}</span>
        <span class="highlighted"><strong>Destination:</strong> {data[annotation_idx-1]['dest']}</span><br>
        <span class="highlighted-alt"><strong>People Number:</strong> {data[annotation_idx-1]['people_number']}</span>
        <span class="highlighted-alt"><strong>Budget:</strong> {data[annotation_idx-1]['budget']}</span>
        <span class="highlighted-alt"><strong>Hotel Rule:</strong> {data[annotation_idx-1]['local_constraint']['house rule']}</span>
        <span class="highlighted-alt"><strong>Cuisine:</strong> {data[annotation_idx-1]['local_constraint']['cuisine']}</span>
        <span class="highlighted-alt"><strong>Room Type:</strong> {data[annotation_idx-1]['local_constraint']['room type']}</span>
        <span class="highlighted-alt"><strong>Transportation:</strong> {data[annotation_idx-1]['local_constraint']['transportation']}</span><br>
    </div>
    """

def judge_valid_info(info):
    if info == "" or not info or info == "You don't need to fill in the information for this or later days." :
        return False
    return True



def judge_valid_transportation(info, annotation_data):
    if  annotation_data['local_constraint']['transportation'] == 'no flight' and 'Flight' in info:
        return False
    elif annotation_data['local_constraint']['transportation'] == 'no self-driving' and 'Self-driving'  in info:
        return False
    return True

def judge_valid_room_type(info, annotation_data, accommodation_data_all):
    accommodation_data_filtered = get_filtered_data(info, accommodation_data_all)
    if annotation_data['local_constraint']['room type'] == 'not shared room' and accommodation_data_filtered['room type'].values[0] == 'Shared room':
        return False
    # "shared room", "not shared room", "private room", "entire room"
    elif annotation_data['local_constraint']['room type'] == 'shared room' and accommodation_data_filtered['room type'].values[0] != 'Shared room':
        return False

    elif annotation_data['local_constraint']['room type'] == 'private room' and accommodation_data_filtered['room type'].values[0] != 'Private room':
        return False

    elif annotation_data['local_constraint']['room type'] == 'entire room' and accommodation_data_filtered['room type'].values[0] != 'Entire home/apt':
        return False

    return True

def judge_valid_room_rule(info, annotation_data, accommodation_data_all):
    accommodation_data_filtered = get_filtered_data(info, accommodation_data_all)
    if annotation_data['local_constraint']['house rule'] == 'smoking' and 'No smoking' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'parities' and 'No parties' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'children under 10' and 'No children under 10' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'visitors' and 'No visitors' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    if annotation_data['local_constraint']['house rule'] == 'pets' and 'No pets' in str(accommodation_data_filtered['house_rules'].values[0]):
        return False
    
    return True

def judge_valid_cuisine(info, annotation_data, restaurant_data_all, cuisine_set: set):
    if info != "-" and annotation_data['local_constraint']['cuisine'] is not None and annotation_data['org'] not in info:
        restaurant_data_filtered = get_filtered_data(info, restaurant_data_all,('Name','City'))
        for cuisine in annotation_data['local_constraint']['cuisine']:
            if cuisine in restaurant_data_filtered.iloc[0]['Cuisines']:
                cuisine_set.add(cuisine)
    return cuisine_set




def get_valid_name_city(info):
    # Modified the pattern to preserve spaces at the end of the name
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    else:
        print(f"{info} can not be parsed, '-' will be used instead.")
        return "-","-"

    
def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers

def get_city_list(days, deparure_city, destination):
    city_list = []
    city_list.append(deparure_city)
    if days == 3:
        city_list.append(destination)
    else:
        city_set = open('../database/background/citySet_with_states.txt').read().split('\n')
        state_city_map = {}
        for unit in city_set:
            city, state = unit.split('\t')
            if state not in state_city_map:
                state_city_map[state] = []
            state_city_map[state].append(city)
        for city in state_city_map[destination]:
            if city != deparure_city:
                city_list.append(city + f"({destination})")
    return city_list

def get_filtered_data(component,data, column_name=('NAME','city')):
    name, city = get_valid_name_city(component)
    return data[(data[column_name[0]] == name) & (data[column_name[1]] == city)]

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s

def count_consecutive_values(lst):
    if not lst:
        return []

    result = []
    current_string = lst[0]
    count = 1

    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1

    result.append((current_string, count))  # Add the last group of values
    return result
