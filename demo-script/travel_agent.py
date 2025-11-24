# -*- coding: utf-8 -*-
"""
Travel Planner Workflow - 旅行规划智能体工作流
集成了 Agent、工具、奖励函数于一个文件中
参考 learn_to_ask 的结构设计
"""

from __future__ import annotations

 
import asyncio
import csv
 
 
from pathlib import Path
from typing import List, Optional, Dict, Set, Union
from dataclasses import dataclass
import os
 
 
 
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeChatFormatter

from agentscope.model import DashScopeChatModel
 
from agentscope.message import Msg

 
 
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
    
    def __init__(self, model,max_iters):
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

        # 创建模型
        self.model = DashScopeChatModel(
            model_name=model,
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            stream=True,
            enable_thinking=True
        )
        
        
        # 创建 Agent
        self.agent = ReActAgent(
            name="travel-planner",
            sys_prompt=AGENT_SYSTEM_PROMPT,
            model=self.model,
            toolkit=toolkit,
            memory=InMemoryMemory(),
            formatter=DashScopeChatFormatter(),
            max_iters=max_iters
             # 限制迭代次数用于测试
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




async def main():
    """测试主函数"""
    print("="*60)
    agent = TravelAgent(model="qwen-max",max_iters=10)
    query = "Please plan a trip for me starting from Sarasota to Chicago for 3 days, from March 22nd to March 24th, 2022. The budget for this trip is set at $1,900."
    
    response = await agent.agent.reply(
            msg=Msg("user", query, role="user"),
        )



    print("="*60+"response"+"="*60) 
    print(response.content)



if __name__ == "__main__":
    asyncio.run(main())