import string
from mip_check import MIPCheck

from helper_function import Helper
from file_reader import get_parameters
from collections import deque
import time
from OSM import build_tokyo_small_parameters_compatible
class Heuristic:
    def __init__(self, parameters):
        """
        Take the parameter to initiate a helper instance
        :param parameters: parameter dict of a graph instance
        """
        self.parameters = parameters
        #self.checker = MIPCheck(self.parameters)
        #self.SI = StationInsertion(self.parameters)
        #self.helper = Helper(self.parameters)
        self.clients = self.parameters["clients"]
        self.stations = self.parameters["stations"]
        self.all_nodes = self.parameters["all_nodes"]
        self.depot_start = self.parameters["depot_start"]
        self.depot_end = self.parameters["depot_end"]
        self.demand = self.parameters["demand"]
        self.ready_time = self.parameters["ready_time"]
        self.due_date = self.parameters["due_date"]
        self.service_time = self.parameters["service_time"]
        self.arcs = self.parameters["arcs"]
        self.times = self.parameters["times"]
        self.final_data = self.parameters["final_data"]
        self.original_stations = self.parameters["original_stations"]
        self.drone_capacity = 50
        self.Q = self.parameters["Q"]
        self.C = self.parameters["C"]
        self.g = self.parameters["g"]
        self.h = self.parameters["h"]
        self.v = self.parameters["v"]
        self.assigned_clients = set()  # 已被无人机服务的客户集合
        self.drone_count = {station: 3 for station in self.stations}  # 每个充电站的可用无人机数量
    def feasible_merge(self, c1, c2, routes):

        route1 = None
        route2 = None

        #
        for route in routes:
            if c1 in route:
                route1 = route
            if c2 in route:
                route2 = route

        # 如果任一客户不在路径中，或在同一路径中，不能合并
        if route1 is None or route2 is None or route1 == route2:
            return False

        # 合并后路径的总需求
        total_demand = 0
        for node in route1 + route2:
            if node in self.clients:
                total_demand += self.demand[node]

        # 判断是否满足容量约束
        return total_demand <= self.C

    def merge_routes(self, c1, c2, routes):

        new_routes = [route.copy() for route in routes]  # 拷贝原路径列表

        route1 = None
        route2 = None

        # 找出包含 c1 和 c2 的路径
        for route in new_routes:
            if c1 in route:
                route1 = route
            if c2 in route:
                route2 = route

        # 如果任一客户不在路径中，或在同一路径中，不进行合并
        if route1 is None or route2 is None or route1 == route2:
            return new_routes

        # 合并路径：route1[:-1] + route2[1:]（去掉重复的 D0_end）
        merged_route = route1[:-1] + route2[1:]

        # 替换原路径
        new_routes.remove(route1)
        new_routes.remove(route2)
        new_routes.append(merged_route)

        return new_routes
    def cvrp_initial_solution(self):

        # Step 1: 初始化每辆车单独服务一个客户
        routes = [
            [self.depot_start[0], client, self.depot_end[0]]  # 提取嵌套列表中的字符串
            for client in self.clients
        ]

        # Step 2: 计算节省量矩阵（Savings Matrix）
        savings = []
        for i in range(len(self.clients)):
            for j in range(i + 1, len(self.clients)):
                c1, c2 = self.clients[i], self.clients[j]
                save = self.arcs[self.depot_start[0], c1] + self.arcs[self.depot_start[0], c2] - self.arcs[c1, c2]
                savings.append((save, c1, c2))
        # Step 3: 按节省量排序并合并路径
        savings.sort(reverse=True)

        for save, c1, c2 in savings:
            if self.feasible_merge(c1, c2, routes):
                routes = self.merge_routes(c1, c2, routes)

        return routes

    def assign_clients_to_stations(self):
        """
        将客户分配到最近的可覆盖它的充电站，考虑无人机数量限制
        """
        self.station_coverage = {}
        self.station_clients = {}

        for client in self.clients:
            reachable_stations = []
            for station in self.stations:
                reachable_stations.append(station)

            if reachable_stations:
                best_station = min(
                    reachable_stations,
                    key=lambda s: self.arcs[s, client]
                )
                self.station_coverage.setdefault(best_station, []).append(client)
                self.station_clients[client] = best_station

    def calculate_drone_savings(self, routes):

        savings = []

        for route_idx, route in enumerate(routes):
            for i in range(1, len(route) - 1):  # 遍历客户节点
                client = route[i]

                if client in self.assigned_clients:
                    continue  # 已被服务
                if self.demand[client] > self.drone_capacity:
                    continue

                if client in self.station_clients:
                    station = self.station_clients[client]

                    if self.drone_count[station] <= 0:
                        continue  # 无人机已用完

                    # 计算节省量
                    direct_cost = self.arcs[route[i - 1], client] + self.arcs[client, route[i + 1]]
                    drone_cost = (
                            self.arcs[route[i - 1], station] +
                            self.arcs[station, client] * 2 +  # 暂时算往返
                            self.arcs[client, route[i + 1]]
                    )
                    save = direct_cost - drone_cost
                    savings.append((save, client, station, route_idx, i))

        return sorted(savings, reverse=True)

    def greedy_insert_drone(self, routes):

        drone_routes = {s: [] for s in self.stations}

        while True:
            # Step 1: 计算当前所有客户的节省量
            savings = self.calculate_drone_savings(routes)

            if not savings:
                break  # 无节省量可插入

            # Step 2: 按节省量排序，插入最优项
            savings.sort(reverse=True)
            best_save = savings[0]
            save, client, station, route_idx, pos = best_save

            if self.drone_count[station] <= 0 or client in self.assigned_clients:
                continue  # 无人机用尽或客户已服务

            # Step 3: 替换路径
            route = routes[route_idx]
            new_route = route[:pos] + [station] + route[pos + 1:]
            if self.is_route_feasible(new_route):
                routes[route_idx] = new_route
                drone_routes[station].append([station, client, station])
                self.assigned_clients.add(client)
                self.drone_count[station] -= 1

                # Step 4: 局部搜索优化路径
                #routes = self.optimize_with_local_search(routes)

        return routes, drone_routes

    def split_paths(self, truck_routes, drone_routes):

        all_paths = []

        # Step 1: 添加卡车路径
        for route in truck_routes:
            all_paths.append(route)

        # Step 2: 添加所有无人机路径
        for station in drone_routes:
            for drone_route in drone_routes[station]:
                all_paths.append(drone_route)

        return all_paths

    def calculate_total_cost(self, truck_routes, drone_routes):

        total_cost = 0.0
        all_paths = self.split_paths(truck_routes, drone_routes)

        for path in all_paths:
            try:
                total_cost += self.calculate_total_distance(path)
            except ValueError as e:
                print(f"路径 {path} 无效: {e}")

        return total_cost
    def calculate_total_distance(self, route):

        total_distance = 0.0
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i + 1]
            if (u, v) in self.arcs:
                total_distance += self.arcs[(u, v)]
            else:
                raise ValueError(f"路径断裂：{u} 到 {v} 之间无有效弧")
        return total_distance
    def is_route_feasible(self, route):

        total_demand = 0
        for node in route:
            if node in self.clients:
                total_demand += self.demand[node]
        return total_demand <= self.C  # 判断是否超过卡车最大载重

    def is_path_continuous(self, route):

        if route[0] != self.depot_start or route[-1] != self.depot_end:
            return False


        visited = set()
        for node in route[1:-1]:  # 忽略起点和终点
            if node in self.stations:
                if node in visited:
                    return False  # 重复访问站点
                visited.add(node)


        for i in range(len(route) - 1):
            if (route[i], route[i + 1]) not in self.arcs:
                return False  # 路径断裂，无法通行

        return True

    def remove_duplicate_stations(self, route):

        new_route = []
        visited_stations = set()

        for node in route:
            if node in self.stations:
                if node in visited_stations:
                    continue  # 跳过重复站点
                visited_stations.add(node)
            new_route.append(node)

        return new_route

    def fix_disconnected_path(self, route):

        new_route = [route[0]]  # 起点
        for i in range(1, len(route)):
            prev = new_route[-1]
            curr = route[i]
            if (prev, curr) in self.arcs:
                new_route.append(curr)
            else:
                # 路径断裂，尝试插入最短路径
                path = self.find_shortest_path(prev, curr)
                if path:
                    new_route.extend(path[1:])  # 跳过起点
                else:
                    return None  # 无法修复
        return new_route

    def find_shortest_path(self, start, end):

        visited = set()
        queue = deque([[start]])

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == end:
                return path
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.all_nodes:
                if (node, neighbor) in self.arcs and neighbor not in visited:
                    queue.append(path + [neighbor])
        return None  # 无路径

    def optimize_with_local_search(self, routes):

        iteration = 0
        while iteration < 3:  # 最多迭代3次
            iteration += 1
            print(f"Local search iteration {iteration}")

            # 1. 2-opt 优化每条路径
            for i in range(len(routes)):
                routes[i] = self.two_opt(routes[i])

            # 2. Swap 优化路径间客户分配
            routes = self.swap(routes)

            # 3. Shift 优化客户重分配
            routes = self.shift(routes)

        return routes

    def two_opt(self, route):
        """
        对单条路径进行 2-opt 优化
        """
        best_route = route.copy()
        improved = True

        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # 忽略充电站作为客户节点
                    if route[i] in self.stations or route[j] in self.stations:
                        continue

                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if self.is_route_feasible(new_route) and self.route_cost(new_route) < self.route_cost(
                            best_route):
                        best_route = new_route
                        improved = True
            route = best_route

        return best_route

    def swap(self, routes):

        best_routes = [r.copy() for r in routes]
        improved = True

        while improved:
            improved = False
            for r1 in range(len(routes)):
                for r2 in range(r1 + 1, len(routes)):
                    for i in range(1, len(routes[r1]) - 1):
                        c1 = routes[r1][i]
                        if c1 in self.assigned_clients:  # 已被无人机服务
                            continue

                        for j in range(1, len(routes[r2]) - 1):
                            c2 = routes[r2][j]
                            if c2 in self.assigned_clients:
                                continue

                            new_r1 = routes[r1][:i] + [c2] + routes[r1][i + 1:]
                            new_r2 = routes[r2][:j] + [c1] + routes[r2][j + 1:]

                            if (self.is_route_feasible(new_r1) and
                                    self.helper.feasible_route(new_r2) and
                                    self.route_cost(new_r1) + self.route_cost(new_r2) <
                                    self.route_cost(routes[r1]) + self.route_cost(routes[r2])):
                                best_routes[r1] = new_r1
                                best_routes[r2] = new_r2
                                improved = True
            routes = best_routes

        return best_routes

    def shift(self, routes):

        best_routes = [r.copy() for r in routes]
        improved = True

        while improved:
            improved = False
            for r1 in range(len(routes)):
                for i in range(1, len(routes[r1]) - 1):
                    c = routes[r1][i]
                    if c in self.assigned_clients:  # 已被无人机服务
                        continue

                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        for j in range(1, len(routes[r2]) - 1):
                            new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                            new_r2 = routes[r2][:j] + [c] + routes[r2][j:]

                            if (self.is_route_feasible(new_r2) and
                                    self.route_cost(new_r1) + self.route_cost(new_r2) <
                                    self.route_cost(routes[r1]) + self.route_cost(routes[r2])):
                                best_routes[r1] = new_r1
                                best_routes[r2] = new_r2
                                improved = True
            routes = best_routes

        return best_routes

    def route_cost(self, route):

        return sum(self.arcs[route[i], route[i + 1]] for i in range(len(route) - 1))

    def hybrid_optimized_solution(self):

        self.assigned_clients = set()
        #self.drone_count = {station: 3 for station in self.stations}
        truck_routes = self.cvrp_initial_solution()

        # Step 2: 预分配客户到充电站（考虑无人机数量限制）
        self.assign_clients_to_stations()

        # Step 3: 贪心插入无人机
        truck_routes, drone_routes = self.greedy_insert_drone(truck_routes)

        return truck_routes, drone_routes




