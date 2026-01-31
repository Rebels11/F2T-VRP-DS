from collections import deque

class GreedyInitial:
    def __init__(self, parameters):

        self.parameters = parameters
        # 直接对齐你现有字段命名，便于复用
        self.clients = parameters["clients"]
        self.stations = parameters["stations"]
        self.all_nodes = parameters["all_nodes"]
        self.depot_start = parameters["depot_start"]   # 形如 ["D0"]
        self.depot_end = parameters["depot_end"]       # 形如 ["D0_end"]
        self.demand = parameters["demand"]
        self.arcs = parameters["arcs"]
        self.C = parameters["C"]

        # 简单设置：每个站点 1 架无人机（与您代码中保持一致的默认）
        self.drone_count = {s: 1 for s in self.stations}

        # 记录无人机分配：client -> station
        self.station_clients = {}
        # 已被无人机服务的客户集合（用于避免再由卡车服务）
        self.assigned_clients = set()

    # ---------------- 基础工具函数 ----------------
    def feasible_add(self, current_load, client):
        """判断将 client 加入当前卡车是否超容量"""
        return (current_load + self.demand[client]) <= self.C

    def route_cost(self, route):
        """计算路径总路程"""
        return sum(self.arcs[(route[i], route[i+1])] for i in range(len(route)-1))

    def calculate_total_distance(self, route):
        """与你现有风格一致的单条路径距离计算"""
        total = 0.0
        for i in range(len(route)-1):
            u, v = route[i], route[i+1]
            if (u, v) not in self.arcs:
                raise ValueError(f"路径断裂：{u}->{v} 不存在有效弧")
            total += self.arcs[(u, v)]
        return total

    # ---------------- 第1步：卡车最近邻贪心 ----------------
    def greedy_truck_routes(self):

        unserved = set(self.clients)
        routes = []

        while unserved:
            route = [self.depot_start[0]]
            load = 0
            last = route[-1]

            while True:
                # 在剩余客户中找最近且可行者
                best_c = None
                best_dist = float('inf')

                for c in unserved:
                    # 需要存在弧 (last, c) 才可行
                    if (last, c) not in self.arcs:
                        continue
                    if not self.feasible_add(load, c):
                        continue
                    dist = self.arcs[(last, c)]
                    if dist < best_dist:
                        best_dist = dist
                        best_c = c

                if best_c is None:
                    # 无可行客户可接，封路
                    route.append(self.depot_end[0])
                    routes.append(route)
                    break
                else:
                    # 加入该客户并继续
                    route.append(best_c)
                    load += self.demand[best_c]
                    last = best_c
                    unserved.remove(best_c)

        return routes


    def assign_clients_to_nearest_station(self):
        """
        为每个客户记录“最近站点”（不考虑续航/能量，仅作为初始贪心替换的候选）
        """
        for c in self.clients:
            # 若没有站点，直接跳过
            if not self.stations:
                continue
            # 选择与客户距离最近的站点（要求弧存在）
            best_s = None
            best_d = float('inf')
            for s in self.stations:
                if (s, c) in self.arcs:
                    d = self.arcs[(s, c)]
                    if d < best_d:
                        best_d = d
                        best_s = s
            if best_s is not None:
                self.station_clients[c] = best_s

    def greedy_one_pass_drone_replace(self, truck_routes):

        drone_routes = {s: [] for s in self.stations}

        for r_idx, route in enumerate(truck_routes):
            # 形如 [D0, c1, c2, ..., D0_end]
            # 我们在 1..len-2 位置遍历客户
            i = 1
            while i < len(route) - 1:
                c = route[i]
                # 若不是客户（例如站点或别的），跳过
                if c not in self.clients:
                    i += 1
                    continue

                # 没有站点候选，跳过
                if c not in self.station_clients:
                    i += 1
                    continue

                s = self.station_clients[c]
                # 该站无人机已用完，跳过
                if self.drone_count.get(s, 0) <= 0:
                    i += 1
                    continue

                prev_node = route[i-1]
                next_node = route[i+1]

                # 必须存在相应弧
                has_prev_c = (prev_node, c) in self.arcs
                has_c_next = (c, next_node) in self.arcs
                has_prev_s = (prev_node, s) in self.arcs
                has_s_next = (s, next_node) in self.arcs
                has_s_c = (s, c) in self.arcs

                if not (has_prev_c and has_c_next and has_prev_s and has_s_next and has_s_c):
                    i += 1
                    continue

                # 卡车直接服务该客户的“局部成本”（prev->c->next）
                direct_cost = self.arcs[(prev_node, c)] + self.arcs[(c, next_node)]
                # 若由无人机服务：卡车改为 prev->s->next；无人机走 s->c->s（这里用“往返”等距假设）
                truck_cost_if_drone = self.arcs[(prev_node, s)] + self.arcs[(s, next_node)]
                drone_cost = self.arcs[(s, c)] + self.arcs[(c, s)]
                delta = direct_cost - (truck_cost_if_drone + drone_cost)

                # 若 delta > 0 则替换更省
                if delta > 0:
                    # 替换：route[i] = s（卡车不再访问 c，而是访问 s）
                    route[i] = s
                    # 记录无人机路径
                    drone_routes[s].append([s, c, s])

                    self.drone_count[s] -= 1
                    self.assigned_clients.add(c)

                    i += 1
                else:
                    i += 1

        return truck_routes, drone_routes

    # ---------------- 对外主入口 ----------------
    def build(self):

        truck_routes = self.greedy_truck_routes()

        # Step 2: 预计算每个客户的最近站点
        self.assign_clients_to_nearest_station()

        # Step 3: 单次无人机替换
        truck_routes, drone_routes = self.greedy_one_pass_drone_replace(truck_routes)

        return truck_routes, drone_routes
