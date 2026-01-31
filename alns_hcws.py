from file_reader import get_parameters
from helper_function import Helper
from Initial_hcws import Heuristic
import random
import math
from time import time
from Initial_hcws import Heuristic
from greedy import GreedyInitial
import time
import matplotlib.pyplot as plt
from OSM import build_tokyo_small_parameters_compatible




class ALNS:
    def __init__(self, parameters):
        self.parameters = parameters
        #self.helper = Helper(self.parameters)
        #self.initial = Heuristic.cvrp_initial_solution()
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
        self.Q = self.parameters["Q"]
        self.C = self.parameters["C"]
        self.g = self.parameters["g"]
        self.h = self.parameters["h"]
        self.v = self.parameters["v"]
        self.assigned_clients = set()  # 已被无人机服务的客户集合
        self.drone_count = {station: 3 for station in self.stations}  # 每个充电站的可用无人机数量
        self.drone_capacity = 50
        # 在 __init__ 中添加以下字段
        self.temperature = 1000.0  # 初始温度
        self.cooling_rate = 0.990  # 降温速率
        self.iteration = 0  # 当前迭代次数
        self.max_iterations = 10000  # 最大迭代次数
        self.no_improvement_count = 0  # 无改进计数器
        self.max_no_improvement = 200  # 最大无改进次数
        # 破坏算子权重初始化
        self.destroy_weights = {
            "random_removal": 1.0,
            "worst_removal": 1.0,
            "shark_removal": 1.0,
            "shaw_removal": 1.0,
            "worst_drone_removal": 1.0,
        }
        # 修复算子权重初始化
        self.repair_weights = {
            "greedy_insert": 1.0,
            "regret_insert": 1.0,
            "best_drone_insertion": 1.0,
        }

    def best_drone_insertion(self, truck_routes, drone_routes, removed_clients):


        for client in list(removed_clients):
            # 当前解基准成本
            base_solution = {"truck": truck_routes, "drones": drone_routes}
            base_cost = self.calculate_total_cost(base_solution)

            # ---- ① 先尝试无人机优先插入 ----
            best_drone_choice = None
            # best_drone_choice: (cost_increase, route_idx_or_None, pos_or_None, station, mode)
            # mode: "add_only" (station already in route) or "insert_station"

            station = self.select_best_station_for_client(client)
            # 如果没有任何可用 station，则直接走卡车插入
            if station is not None:
                # 深拷贝 drones 用于候选评估
                def deepcopy_drones(drs):
                    return {s: [rt[:] for rt in routes] for s, routes in drs.items()}

                # (A) station 已在某条卡车路径中：无需插 station，只加 drone 任务
                for route_idx, route in enumerate(truck_routes):
                    if station in route:
                        cand_truck = [r[:] for r in truck_routes]
                        cand_drones = deepcopy_drones(drone_routes)
                        if station not in cand_drones:
                            cand_drones[station] = []

                        # 无人机数量限制
                        if station in self.drone_count and len(cand_drones[station]) + 1 > self.drone_count[station]:
                            continue

                        # 无人机载重限制
                        if self.demand.get(client, 0) > self.drone_capacity:
                            continue

                        cand_drones[station].append([station, client, station])

                        cand_solution = {"truck": cand_truck, "drones": cand_drones}
                        new_cost = self.calculate_total_cost(cand_solution)
                        inc = new_cost - base_cost

                        if (best_drone_choice is None) or (inc < best_drone_choice[0]):
                            best_drone_choice = (inc, route_idx, None, station, "add_only")

                # (B) 尝试在各条路径的各位置插 station，然后用 drone 服务 client
                for route_idx, route in enumerate(truck_routes):
                    for pos in range(1, len(route)):  # 不插在 depot 前
                        new_route = route[:pos] + [station] + route[pos:]

                        # 卡车容量可行性（你当前只查容量）
                        if not self.is_route_feasible(new_route):
                            continue

                        cand_truck = [r[:] for r in truck_routes]
                        cand_truck[route_idx] = new_route

                        cand_drones = deepcopy_drones(drone_routes)
                        if station not in cand_drones:
                            cand_drones[station] = []

                        # 无人机数量限制
                        if station in self.drone_count and len(cand_drones[station]) + 1 > self.drone_count[station]:
                            continue

                        # 无人机载重限制
                        if self.demand.get(client, 0) > self.drone_capacity:
                            continue

                        cand_drones[station].append([station, client, station])

                        cand_solution = {"truck": cand_truck, "drones": cand_drones}
                        new_cost = self.calculate_total_cost(cand_solution)
                        inc = new_cost - base_cost

                        if (best_drone_choice is None) or (inc < best_drone_choice[0]):
                            best_drone_choice = (inc, route_idx, pos, station, "insert_station")

            # 若无人机方案存在，则直接采用（drone-priority）
            if best_drone_choice is not None:
                _, ridx, pos, station, mode = best_drone_choice

                if mode == "add_only":
                    # truck 不变，仅添加 drone 任务
                    if station not in drone_routes:
                        drone_routes[station] = []
                    drone_routes[station].append([station, client, station])

                else:  # "insert_station"
                    # 在指定路线 pos 插 station + drone 任务
                    route = truck_routes[ridx]
                    truck_routes[ridx] = route[:pos] + [station] + route[pos:]

                    if station not in drone_routes:
                        drone_routes[station] = []
                    drone_routes[station].append([station, client, station])

                continue  # ✅ drone-priority: 成功则不再考虑 truck 插入

            # ---- ② 无无人机可行插入时：退化为卡车插入（best position）----
            best_truck_choice = None  # (inc, route_idx, pos)
            for route_idx, route in enumerate(truck_routes):
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [client] + route[pos:]
                    if not self.is_route_feasible(new_route):
                        continue

                    cand_truck = [r[:] for r in truck_routes]
                    cand_truck[route_idx] = new_route
                    cand_drones = {s: [rt[:] for rt in routes] for s, routes in drone_routes.items()}

                    cand_solution = {"truck": cand_truck, "drones": cand_drones}
                    new_cost = self.calculate_total_cost(cand_solution)
                    inc = new_cost - base_cost

                    if (best_truck_choice is None) or (inc < best_truck_choice[0]):
                        best_truck_choice = (inc, route_idx, pos)

            if best_truck_choice is not None:
                _, ridx, pos = best_truck_choice
                route = truck_routes[ridx]
                truck_routes[ridx] = route[:pos] + [client] + route[pos:]
            else:

                pass

        return truck_routes, drone_routes

    def worst_drone_removal(self, truck_routes, drone_routes, remove_rate=0.3):

        new_truck = [r[:] for r in truck_routes]
        new_drones = {s: [rt[:] for rt in routes] for s, routes in drone_routes.items()}


        drone_tasks = []  # (cost, station, task_index, customer)
        for s, routes in new_drones.items():
            for idx, r in enumerate(routes):
                if len(r) != 3:
                    continue
                _, c, _ = r
                # 计算 station->c->station 的距离
                d1 = self.arcs.get((s, c))
                if d1 is None:
                    d1 = self.arcs.get((c, s), float("inf"))
                d2 = self.arcs.get((c, s))
                if d2 is None:
                    d2 = self.arcs.get((s, c), float("inf"))
                dist = d1 + d2
                cost = dist / 3.0
                drone_tasks.append((cost, s, idx, c))

        if not drone_tasks:
            return new_truck, set(), new_drones


        k = max(1, int(len(drone_tasks) * remove_rate))
        k = min(k, len(drone_tasks))


        drone_tasks.sort(key=lambda x: x[0], reverse=True)
        to_remove = drone_tasks[:k]

        removed_clients = set()


        remove_map = {}  # station -> set(indices)
        for _, s, idx, c in to_remove:
            removed_clients.add(c)
            remove_map.setdefault(s, set()).add(idx)


        for s, idx_set in remove_map.items():
            routes = new_drones.get(s, [])

            for idx in sorted(idx_set, reverse=True):
                if 0 <= idx < len(routes):
                    routes.pop(idx)
            new_drones[s] = routes

        # 删除“起飞站点”在卡车路径中的访问（only if station no longer has any drone task）
        empty_stations = [s for s, routes in new_drones.items() if len(routes) == 0]
        empty_stations_set = set(empty_stations)

        if empty_stations_set:
            updated_truck = []
            for route in new_truck:

                updated_route = [node for node in route if node not in empty_stations_set]
                updated_truck.append(updated_route)
            new_truck = updated_truck


            for s in empty_stations:

                new_drones.pop(s, None)

        return new_truck, removed_clients, new_drones

    def shaw_removal(
            self,
            truck_routes,
            remove_rate=0.3,
            gamma=1.0,
            theta=0.1,
            lam=-5.0
    ):


        gamma = self.sr_gamma if gamma is None else gamma
        theta = self.sr_theta if theta is None else theta
        lam = self.sr_lambda if lam is None else lam

        # 1) 收集当前所有“卡车服务的客户”
        all_clients = []
        client_to_route = {}  # 记录客户所在路径 idx
        for ridx, route in enumerate(truck_routes):
            for node in route:
                if node in self.clients:
                    all_clients.append(node)
                    client_to_route[node] = ridx

        if not all_clients:
            return [r[:] for r in truck_routes], []


        num_to_remove = max(1, int(len(all_clients) * remove_rate))
        num_to_remove = min(num_to_remove, len(all_clients))


        seed = random.choice(all_clients)

        def dist(u, v):

            d = self.arcs.get((u, v))
            if d is None:
                d = self.arcs.get((v, u))
            if d is None:
                return float("inf")
            return d

        def relatedness(i, j):
            d_ij = dist(i, j)
            di = self.demand.get(i, 0)
            dj = self.demand.get(j, 0)
            g_ij = 1 if client_to_route.get(i) == client_to_route.get(j) else -1
            return gamma * d_ij + theta * abs(di - dj) + lam * g_ij


        candidates = []
        for j in all_clients:
            if j == seed:
                continue
            R = relatedness(seed, j)
            candidates.append((R, j))

        candidates.sort(key=lambda x: x[0])

        removed = [seed] + [j for _, j in candidates[: num_to_remove - 1]]

        # 5) 执行删除：只删客户，不删站点
        removed_set = set(removed)
        new_routes = []
        for route in truck_routes:
            new_route = [node for node in route if node not in removed_set]
            new_routes.append(new_route)

        return new_routes, removed

    def count_truck_served_clients(self, solution):

        served_by_truck = set()
        for route in solution.get("truck", []):
            for node in route:
                if node in self.clients:
                    served_by_truck.add(node)
        return len(served_by_truck)
    def alns_algorithm(self, initial_truck_routes, drone_routes):
        """
           Adaptive Large Neighborhood Search (ALNS) 算法主流程
           """
        current_solution = {
            "truck": initial_truck_routes,
            "drones": drone_routes,
        }
        current_solution["cost"] = self.calculate_total_cost(current_solution)

        best_solution = current_solution.copy()
        destroy_operators = list(self.destroy_weights.keys())
        repair_operators = list(self.repair_weights.keys())

        while self.iteration < self.max_iterations and self.no_improvement_count < self.max_no_improvement:
            # Step 1: 随机选择破坏和修复算子
            destroy_op = self.select_operator(destroy_operators, self.destroy_weights)
            repair_op = self.select_operator(repair_operators, self.repair_weights)

            # Step 2: 应用破坏和修复算子
            destroyed = self.apply_destroy(destroy_op,
                                           current_solution["truck"],
                                           current_solution["drones"])
            repaired_truck, repaired_drones = self.apply_repair(
                repair_op, destroyed, current_solution["drones"]
            )

            new_solution = {
                "truck": repaired_truck,
                "drones": repaired_drones,
            }
            new_solution["cost"] = self.calculate_total_cost(new_solution)

            # Step 4: 判断是否接受新解
            if self.accept_solution(current_solution, new_solution):
                current_solution = new_solution
                if new_solution["cost"] < best_solution["cost"]:
                    best_solution = new_solution
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            else:
                self.no_improvement_count += 1

            # Step 5: 更新算子权重
            self.update_operator_weights(destroy_op, repair_op, current_solution, new_solution)

            # Step 6: 降温
            self.temperature *= self.cooling_rate
            self.iteration += 1

        return best_solution

    def random_removal(self, truck_routes, remove_rate=0.3):

        removed_clients = set()
        all_clients = [c for route in truck_routes for c in route if c in self.clients]
        num_to_remove = max(1, int(len(all_clients) * remove_rate))
        num_to_remove = min(num_to_remove, len(all_clients))

        for _ in range(num_to_remove):
            client = random.choice(all_clients)
            for i, route in enumerate(truck_routes):
                if client in route:
                    truck_routes[i].remove(client)
                    removed_clients.add(client)
                    break

        return truck_routes, removed_clients

    def worst_removal(self, truck_routes, remove_rate=0.3):

        savings = []
        for route in truck_routes:
            for i in range(1, len(route) - 1):
                node = route[i]
                if node in self.clients:
                    # 计算移除该客户带来的节省量
                    cost_before = self.calculate_total_distance(route)
                    new_route = route[:i] + route[i + 1:]
                    cost_after = self.calculate_total_distance(new_route)
                    savings.append((cost_before - cost_after, node, i, route))

        savings.sort(reverse=True)  # 从节省量最高的客户开始移除
        removed_clients = set()

        for _, client, _, _ in savings[:int(len(savings) * remove_rate)]:
            for route in truck_routes:
                if client in route:
                    route.remove(client)
                    removed_clients.add(client)
                    break

        return truck_routes, removed_clients

    def shark_removal(self, truck_routes, segment_length=5):

        removed_clients = []
        new_routes = []

        for route in truck_routes:
            # 找出该路径中所有客户的位置
            client_positions = [i for i, node in enumerate(route) if node in self.clients]

            if len(client_positions) == 0:
                # 没有客户，原样保留
                new_routes.append(route[:])
                continue

            # 从客户位置中选一个起点
            start_pos_idx = random.randint(0, len(client_positions) - 1)
            start = client_positions[start_pos_idx]
            # 取 segment_length 个客户位置
            end_pos_idx = min(start_pos_idx + segment_length, len(client_positions))
            seg_positions = client_positions[start_pos_idx:end_pos_idx]

            # 记录被移除的客户
            removed_clients.extend(route[i] for i in seg_positions)

            # 删除这些客户，站点和 depot 不动
            new_route = [node for idx, node in enumerate(route) if idx not in seg_positions]
            new_routes.append(new_route)

        return new_routes, removed_clients

    def greedy_insert(self, truck_routes, drone_routes, removed_clients):


        for client in removed_clients:
            best_choice = None

            base_solution = {
                "truck": truck_routes,
                "drones": drone_routes,
            }
            base_cost = self.calculate_total_cost(base_solution)


            station_for_client = self.select_best_station_for_client(client)

            for route_idx, route in enumerate(truck_routes):


                for pos in range(1, len(route)):  # 不能插在 depot 前面

                    new_route_truck = route[:pos] + [client] + route[pos:]

                    # 容量检查
                    if not self.is_route_feasible(new_route_truck):
                        continue

                    # 2) 构造候选解（卡车改了，无人机不变）
                    candidate_truck = [r[:] for r in truck_routes]
                    candidate_truck[route_idx] = new_route_truck

                    candidate_drones = {
                        s: [rt[:] for rt in routes]
                        for s, routes in drone_routes.items()
                    }

                    candidate_solution = {
                        "truck": candidate_truck,
                        "drones": candidate_drones,
                    }
                    new_cost = self.calculate_total_cost(candidate_solution)
                    cost_increase = new_cost - base_cost

                    if (best_choice is None) or (cost_increase < best_choice[0]):
                        best_choice = (cost_increase, "truck", route_idx, pos, None)


                if station_for_client is not None:
                    station = station_for_client
                    for pos in range(1, len(route)):
                        # 1) 在 pos 插入 station
                        new_route_drone = route[:pos] + [station] + route[pos:]

                        if not self.is_route_feasible(new_route_drone):
                            continue

                        # 2) 构造候选解
                        candidate_truck = [r[:] for r in truck_routes]
                        candidate_truck[route_idx] = new_route_drone

                        candidate_drones = {
                            s: [rt[:] for rt in routes]
                            for s, routes in drone_routes.items()
                        }
                        if station not in candidate_drones:
                            candidate_drones[station] = []

                        # 添加无人机任务 station -> client -> station
                        candidate_drones[station].append([station, client, station])

                        # 无人机数量上限检查
                        if station in self.drone_count and len(candidate_drones[station]) > self.drone_count[station]:
                            continue


                        candidate_solution = {
                            "truck": candidate_truck,
                            "drones": candidate_drones,
                        }
                        new_cost = self.calculate_total_cost(candidate_solution)
                        cost_increase = new_cost - base_cost

                        if (best_choice is None) or (cost_increase < best_choice[0]):
                            best_choice = (cost_increase, "drone", route_idx, pos, station)


            if best_choice is not None:
                _, mode, route_idx, pos, station = best_choice
                route = truck_routes[route_idx]

                if mode == "truck":
                    # 卡车直接服务该客户
                    truck_routes[route_idx] = route[:pos] + [client] + route[pos:]
                    # 无人机不变

                elif mode == "drone":
                    # 在卡车路径中插入 station，同时建无人机路径
                    truck_routes[route_idx] = route[:pos] + [station] + route[pos:]
                    if station not in drone_routes:
                        drone_routes[station] = []
                    drone_routes[station].append([station, client, station])
            else:

                continue

        return truck_routes, drone_routes

    def regret_insert(self, truck_routes, drone_routes, removed_clients):


        for client in removed_clients:

            choices = []


            base_solution = {
                "truck": truck_routes,
                "drones": drone_routes,
            }
            base_cost = self.calculate_total_cost(base_solution)


            station_for_client = self.select_best_station_for_client(client)


            for route_idx, route in enumerate(truck_routes):

                insertion_options = []  # 存储 (cost_increase, mode, pos, station)


                for pos in range(1, len(route)):  # 不能插在 depot 前面
                    new_route_truck = route[:pos] + [client] + route[pos:]

                    if not self.is_route_feasible(new_route_truck):
                        continue

                    candidate_truck = [r[:] for r in truck_routes]
                    candidate_truck[route_idx] = new_route_truck

                    candidate_drones = {
                        s: [rt[:] for rt in routes]
                        for s, routes in drone_routes.items()
                    }

                    candidate_solution = {
                        "truck": candidate_truck,
                        "drones": candidate_drones,
                    }
                    new_cost = self.calculate_total_cost(candidate_solution)
                    cost_increase = new_cost - base_cost

                    insertion_options.append((cost_increase, "truck", pos, None))


                if station_for_client is not None:
                    station = station_for_client
                    for pos in range(1, len(route)):
                        new_route_drone = route[:pos] + [station] + route[pos:]

                        if not self.is_route_feasible(new_route_drone):
                            continue

                        candidate_truck = [r[:] for r in truck_routes]
                        candidate_truck[route_idx] = new_route_drone

                        candidate_drones = {
                            s: [rt[:] for rt in routes]
                            for s, routes in drone_routes.items()
                        }
                        if station not in candidate_drones:
                            candidate_drones[station] = []

                        candidate_drones[station].append([station, client, station])


                        if station in self.drone_count and len(candidate_drones[station]) > self.drone_count[station]:
                            continue



                        candidate_solution = {
                            "truck": candidate_truck,
                            "drones": candidate_drones,
                        }
                        new_cost = self.calculate_total_cost(candidate_solution)
                        cost_increase = new_cost - base_cost

                        insertion_options.append((cost_increase, "drone", pos, station))

                #
                if insertion_options:

                    insertion_options.sort(key=lambda x: x[0])
                    best_cost, best_mode, best_pos, best_station = insertion_options[0]

                    if len(insertion_options) == 1:
                        second_cost = best_cost
                    else:
                        second_cost = insertion_options[1][0]

                    regret = second_cost - best_cost
                    choices.append(
                        (regret, best_cost, best_mode, best_pos, best_station, route_idx)
                    )

            #
            if choices:
                # 先按 regret 降序，再按 best_cost 升序
                choices.sort(key=lambda x: (-x[0], x[1]))
                _, _, best_mode, best_pos, best_station, route_idx = choices[0]

                route = truck_routes[route_idx]

                if best_mode == "truck":
                    # 卡车直接插入 client
                    truck_routes[route_idx] = route[:best_pos] + [client] + route[best_pos:]
                elif best_mode == "drone":
                    # 插 station + 建无人机路径
                    station = best_station
                    truck_routes[route_idx] = route[:best_pos] + [station] + route[best_pos:]
                    if station not in drone_routes:
                        drone_routes[station] = []
                    drone_routes[station].append([station, client, station])
            else:
                # 该 client 在所有路径上都插不进去（比较少见），暂时忽略
                continue

        return truck_routes, drone_routes

    def drone_insert(self, truck_routes, removed_clients):

        for client in removed_clients:
            if client in self.clients:
                station = self.stations[client]
                if self.drone_count[station] > 0:
                    # 插入到卡车路径中最近的站点
                    for i, route in enumerate(truck_routes):
                        for j in range(1, len(route) - 1):
                            if route[j] in self.stations and route[j] == station:
                                truck_routes[i] = route[:j] + [client] + route[j:]
                                self.drone_count[station] -= 1
                                removed_clients.remove(client)
                                break

        return truck_routes

    def select_operator(self, operators, weights):

        total = sum(weights[op] for op in operators)
        rand = random.uniform(0, total)
        current = 0

        for op in operators:
            current += weights[op]
            if current > rand:
                return op

    def accept_solution(self, current, new):

        delta = new["cost"] - current["cost"]
        if delta < 0:
            return True
        elif self.temperature > 0:
            return random.random() < math.exp(-delta / self.temperature)
        return False

    def update_operator_weights(self, destroy_op, repair_op, current, new):

        delta = new["cost"] - current["cost"]
        if delta < 0:
            score = 1.0
        elif delta == 0:
            score = 0.5  #
        else:
            score = 0.0  #

        # 更新破坏算子权重
        self.destroy_weights[destroy_op] = max(0.1, self.destroy_weights[destroy_op] + score)

        # 更新修复算子权重
        self.repair_weights[repair_op] = max(0.1, self.repair_weights[repair_op] + score)

    def apply_destroy(self, destroy_op, truck_routes, drone_routes=None):

        if destroy_op == "random_removal":
            return self.random_removal([route[:] for route in truck_routes])  # 深拷贝避免修改原解
        elif destroy_op == "worst_removal":
            return self.worst_removal([route[:] for route in truck_routes])
        elif destroy_op == "shark_removal":
            return self.shark_removal([route[:] for route in truck_routes])
        elif destroy_op == "shaw_removal":  # ✅新增
            return self.shaw_removal([route[:] for route in truck_routes])
        elif destroy_op == "worst_drone_removal":
            if drone_routes is None:
                drone_routes = {}
            return self.worst_drone_removal([r[:] for r in truck_routes],
                                            {s: [rt[:] for rt in routes] for s, routes in drone_routes.items()})
        else:
            raise ValueError(f"未知的破坏算子: {destroy_op}")

    def select_best_station_for_client(self, client):

        best_station = None
        best_dist = float("inf")

        for s in self.stations:
            d1 = self.arcs.get((s, client), float("inf"))
            d2 = self.arcs.get((client, s), float("inf"))
            if d1 == float("inf") or d2 == float("inf"):
                continue
            if d1 + d2 < best_dist:
                best_dist = d1 + d2
                best_station = s

        return best_station

    def insert_station_and_drone(self, route, drone_routes, station, client, pos):

        new_route = route[:pos] + [station] + route[pos:]


        if station not in drone_routes:
            drone_routes[station] = []


        drone_routes[station].append([station, client, station])

        return new_route, drone_routes

    def apply_repair(self, repair_op, destroyed_result, current_drones):

        if len(destroyed_result) == 2:
            truck_routes, removed_clients = destroyed_result
            drone_routes = {s: [rt[:] for rt in routes] for s, routes in current_drones.items()}
        else:
            truck_routes, removed_clients, drone_routes = destroyed_result
            truck_routes = [r[:] for r in truck_routes]
            drone_routes = {s: [rt[:] for rt in routes] for s, routes in drone_routes.items()}

        truck_routes = [r[:] for r in truck_routes]


        truck_routes, drone_routes, extra_removed = self._enforce_station_once(truck_routes, drone_routes)
        removed_clients = set(removed_clients) | set(extra_removed)

        if removed_clients:
            if repair_op == "greedy_insert":
                truck_routes, drone_routes = self.greedy_insert(truck_routes, drone_routes, removed_clients)
            elif repair_op == "regret_insert":
                truck_routes, drone_routes = self.regret_insert(truck_routes, drone_routes, removed_clients)
            elif repair_op == "best_drone_insertion":
                truck_routes, drone_routes = self.best_drone_insertion(truck_routes, drone_routes, removed_clients)
            else:
                raise ValueError(f"未知的修复算子: {repair_op}")


        truck_routes, drone_routes, extra_removed2 = self._enforce_station_once(truck_routes, drone_routes)
        if extra_removed2:
            # 把因 station-once 导致的“无人机客户失效”再插回去
            truck_routes, drone_routes = self.best_drone_insertion(truck_routes, drone_routes, list(extra_removed2))


        truck_routes, drone_routes = self._repair_global_service(truck_routes, drone_routes)

        return truck_routes, drone_routes

    def calculate_total_cost(self, solution):

        total_distance = 0.0

        # === 1️⃣ 卡车路径 ===
        for route in solution.get('truck', []):
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                d = self.arcs.get((u, v))
                if d is None:

                    u_fixed = u.replace("_end", "")
                    v_fixed = v.replace("_end", "")
                    d = self.arcs.get((u_fixed, v_fixed), float("inf"))
                    d = d
                total_distance += d/1

        # === 2️⃣ 无人机路径 ===
        for station, drone_routes in solution.get('drones', {}).items():
            for route in drone_routes:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    d = self.arcs.get((u, v))
                    if d is None:
                        #
                        d = self.arcs.get((v, u), float("inf"))
                        d = d
                    total_distance += d/2

        return total_distance

    def _used_stations(self, truck_routes):

        used = set()
        for r in truck_routes:
            for n in r:
                if n in self.stations:
                    used.add(n)
        return used

    def _station_once_ok(self, truck_routes):

        seen = set()
        for r in truck_routes:
            for n in r:
                if n in self.stations:
                    if n in seen:
                        return False
                    seen.add(n)
        return True

    def _enforce_station_once(self, truck_routes, drone_routes):

        removed_clients = set()

        seen = set()
        new_truck = []
        removed_stations = set()

        for route in truck_routes:
            nr = []
            for n in route:
                if n in self.stations:
                    if n in seen:
                        # 后续出现的 station 直接删掉
                        removed_stations.add(n)
                        continue
                    seen.add(n)
                nr.append(n)
            new_truck.append(nr)


        used_after = self._used_stations(new_truck)

        new_drones = {s: [rt[:] for rt in routes] for s, routes in drone_routes.items()}
        for s in list(new_drones.keys()):
            if s not in used_after:
                #
                for r in new_drones[s]:
                    if len(r) == 3:
                        _, c, _ = r
                        removed_clients.add(c)
                new_drones.pop(s, None)

        return new_truck, new_drones, removed_clients

    def _repair_global_service(self, truck_routes, drone_routes):

        served_by_truck = set()
        for r in truck_routes:
            for n in r:
                if n in self.clients:
                    served_by_truck.add(n)

        # 统计 drone 服务
        served_by_drone = set()
        for s, routes in drone_routes.items():
            for r in routes:
                if len(r) == 3:
                    _, c, _ = r
                    if c in self.clients:
                        served_by_drone.add(c)

        used_stations = self._used_stations(truck_routes)

        # 1) 处理重复服务：truck + drone 同时服务
        dup = served_by_truck.intersection(served_by_drone)
        if dup:

            for c in list(dup):

                drone_pairs = []
                for s, routes in drone_routes.items():
                    for idx, r in enumerate(routes):
                        if len(r) == 3 and r[1] == c:
                            drone_pairs.append((s, idx))

                #
                can_keep_drone = (self.demand.get(c, 0) <= self.drone_capacity)
                can_keep_drone = can_keep_drone and any(s in used_stations for (s, _) in drone_pairs)

                if can_keep_drone:

                    for ridx, r in enumerate(truck_routes):
                        if c in r:
                            truck_routes[ridx] = [n for n in r if n != c]
                else:
                    #
                    for (s, idx) in sorted(drone_pairs, key=lambda x: x[1], reverse=True):
                        if s in drone_routes and 0 <= idx < len(drone_routes[s]):
                            drone_routes[s].pop(idx)

                    for s in list(drone_routes.keys()):
                        if len(drone_routes[s]) == 0:
                            drone_routes.pop(s, None)

        # 2) 处理未服务客户
        served_by_truck = set()
        for r in truck_routes:
            for n in r:
                if n in self.clients:
                    served_by_truck.add(n)
        served_by_drone = set()
        for s, routes in drone_routes.items():
            for r in routes:
                if len(r) == 3 and r[1] in self.clients:
                    served_by_drone.add(r[1])

        missing = set(self.clients) - served_by_truck - served_by_drone
        if missing:

            truck_routes, drone_routes = self.best_drone_insertion(truck_routes, drone_routes, list(missing))

        return truck_routes, drone_routes

    def calculate_total_distance(self, route):

        distance = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            distance += self.arcs.get((u, v), float("inf"))  # 从参数里取距离矩阵
        return distance

    def is_solution_feasible(self, truck_routes, drone_routes, check_station_once=True):

        if check_station_once and (not self._station_once_ok(truck_routes)):
            return False

        served_counter = {c: 0 for c in self.clients}

        # 1) 卡车路径容量 + 记录服务
        for route in truck_routes:
            if not self.is_route_feasible(route, served_counter):
                return False

        used_stations = self._used_stations(truck_routes)

        # 2) 无人机任务检查
        for station, routes in drone_routes.items():
            # station 必须被卡车访问
            if routes and (station not in used_stations):
                return False

            max_drones = self.drone_count.get(station, 1)
            if len(routes) > max_drones:
                return False

            for r in routes:
                if len(r) != 3:
                    return False
                launch, customer, rendezvous = r
                if launch != station or rendezvous != station:
                    return False
                if customer not in self.clients:
                    return False

                # 无人机载重
                if self.demand.get(customer, 0) > self.drone_capacity:
                    return False

                served_counter[customer] += 1

        # 3) 每个客户恰好一次
        for c in self.clients:
            if served_counter[c] != 1:
                return False

        return True

    def is_route_feasible(self, route, served_counter=None):

        load = 0
        for node in route:
            if node in self.clients:
                load += self.demand[node]
                if load > self.C:  # 超出车辆容量
                    return False
                if served_counter is not None:
                    served_counter[node] += 1
        return True

    def insert_client(self, route, client):

        best_pos = None
        min_increase = float("inf")

        for i in range(1, len(route)):  # 不能插在 depot 前面
            new_route = route[:i] + [client] + route[i:]
            if self.is_route_feasible(new_route):
                increase = self.calculate_total_distance(new_route) - self.calculate_total_distance(route)
                if increase < min_increase:
                    best_pos = i
                    min_increase = increase

        if best_pos is not None:
            return route[:best_pos] + [client] + route[best_pos:]
        else:

            return route[:-1] + [client] + [route[-1]]

    def alns_optimized_solution(self, truck_routes, drone_routes):
        """
        使用 ALNS 算法优化 CVRP 初始路径
        """
        # Step 1: 生成初始 CVRP 路径

        # Step 3: 插入无人机配送
        #truck_routes, drone_routes = Heuristic.hybrid_optimized_solution()

        # Step 4: 使用 ALNS 优化路径
        best_solution = self.alns_algorithm(truck_routes, drone_routes)

        return best_solution





