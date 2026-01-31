from __future__ import annotations
import math
import random
import time
import copy
from typing import Dict, List, Any, Optional, Tuple


# -----------------------------
# Paper-style CH (same as before, but objective = dist/v)
# -----------------------------
class PaperCHForDroneStations:
    def __init__(self, parameters: Dict[str, Any], seed: int = 0,
                 v_truck: float = 1.0, v_drone: float = 3.0,
                 drone_capacity: float = 50.0,
                 drone_count: Optional[Dict[str, int]] = None):
        self.p = parameters
        self.rng = random.Random(seed)

        self.clients = list(parameters["clients"])
        self.stations = list(parameters["stations"])
        self.depot = parameters["depot_start"][0]
        self.depot_end = parameters["depot_end"][0] if isinstance(parameters["depot_end"], list) else parameters["depot_end"]

        self.arcs = parameters["arcs"]
        self.demand = parameters["demand"]
        self.Q = parameters["Q"]

        self.v_truck = float(v_truck)
        self.v_drone = float(v_drone)

        self.drone_capacity = float(drone_capacity)
        self.drone_count = drone_count if drone_count is not None else {s: 3 for s in self.stations}

        self.node_xy = self._build_xy(parameters["final_data"])

    def build(self, K: Optional[int] = None) -> Dict[str, Any]:
        if K is None:
            K = self._infer_truck_count()

        clusters_kept, overflow = self._cluster_first_corrective(K)

        truck_routes: List[List[str]] = []
        for kept in clusters_kept:
            r = [self.depot, self.depot_end]
            for c in kept:
                r = self._cheapest_insert_node(r, c, speed=self.v_truck)
            truck_routes.append(r)

        N = overflow[:]
        while N:
            h = self.rng.choice(N)
            best = None  # (delta_time, ridx, new_route)
            for ridx, r in enumerate(truck_routes):
                if self._route_load(r) + self.demand[h] > self.Q:
                    continue
                new_r, delta = self._best_insertion_in_route(r, h, speed=self.v_truck)
                if best is None or delta < best[0]:
                    best = (delta, ridx, new_r)
            if best is None:
                truck_routes.append([self.depot, h, self.depot_end])
            else:
                _, ridx, new_r = best
                truck_routes[ridx] = new_r
            N.remove(h)

        drones: Dict[str, List[List[str]]] = {s: [] for s in self.stations}
        truck_routes, drones = self._drone_operations(truck_routes, drones)

        sol = {"truck": truck_routes, "drones": {s: rs for s, rs in drones.items() if rs}}
        sol["cost"] = self.total_time(sol)
        return sol

    def total_time(self, sol: Dict[str, Any]) -> float:
        total = 0.0
        for r in sol.get("truck", []):
            for i in range(len(r) - 1):
                total += self._dist(r[i], r[i + 1]) / self.v_truck
        for st, routes in sol.get("drones", {}).items():
            for dr in routes:
                for i in range(len(dr) - 1):
                    total += self._dist(dr[i], dr[i + 1]) / self.v_drone
        return total

    # --- CH internals ---
    def _cluster_first_corrective(self, K: int):
        customers = self.clients[:]
        clusters, centroids = self._kmeans_customers(customers, K)
        overflow: List[str] = []
        clusters_kept: List[List[str]] = []

        for k_idx in range(len(centroids)):
            Lk = clusters[k_idx]
            if not Lk:
                clusters_kept.append([])
                continue

            def p_w(c: str) -> float:
                wx, wy = self.node_xy[c]
                cx, cy = centroids[k_idx]
                d_self = math.hypot(wx - cx, wy - cy)
                denom = 0.0
                for kk, (cx2, cy2) in enumerate(centroids):
                    if kk == k_idx:
                        continue
                    denom += math.hypot(wx - cx2, wy - cy2)
                if denom <= 1e-12:
                    return float("inf")
                return 2.0 * d_self / denom

            Lk_sorted = sorted(Lk, key=p_w)
            kept, load = [], 0.0
            for c in Lk_sorted:
                d = self.demand[c]
                if load + d <= self.Q:
                    kept.append(c)
                    load += d
                else:
                    overflow.append(c)
            clusters_kept.append(kept)

        return clusters_kept, overflow

    def _best_insertion_in_route(self, route: List[str], node: str, speed: float):
        best_delta = float("inf")
        best_pos = 1
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            delta_dist = self._dist(u, node) + self._dist(node, v) - self._dist(u, v)
            delta_time = delta_dist / speed
            if delta_time < best_delta:
                best_delta = delta_time
                best_pos = i + 1
        new_r = route[:best_pos] + [node] + route[best_pos:]
        return new_r, best_delta

    def _cheapest_insert_node(self, route: List[str], node: str, speed: float):
        new_r, _ = self._best_insertion_in_route(route, node, speed=speed)
        return new_r

    def _route_load(self, route: List[str]) -> float:
        load = 0.0
        cset = set(self.clients)
        for n in route:
            if n in cset:
                load += self.demand[n]
        return load

    def _drone_operations(self, truck_routes, drones):
        # keep it simple (same as earlier): greedy convert if improves
        cset = set(self.clients)

        def rebuild_loc():
            loc = {}
            for ridx, r in enumerate(truck_routes):
                for i, n in enumerate(r):
                    if n in cset:
                        loc[n] = (ridx, i)
            return loc

        loc = rebuild_loc()
        cand_list = []
        for c in self.clients:
            if c not in loc:
                continue
            ridx, pos = loc[c]
            r = truck_routes[ridx]
            if pos <= 0 or pos >= len(r) - 1:
                continue
            prev_n, next_n = r[pos - 1], r[pos + 1]
            saving = (self._dist(prev_n, c) + self._dist(c, next_n) - self._dist(prev_n, next_n)) / self.v_truck
            st, drone_time = self._best_station_roundtrip_time(c)
            if st is None:
                continue
            if self.demand[c] > self.drone_capacity:
                continue
            if len(drones.get(st, [])) >= self.drone_count.get(st, 1):
                continue
            penalty = 0.0 if st in r else self._best_station_insertion_delta_time(r, st)
            benefit = saving - (drone_time + penalty)
            cand_list.append((benefit, c, st))

        cand_list.sort(reverse=True, key=lambda x: x[0])

        cur = {"truck": truck_routes, "drones": {s: rs[:] for s, rs in drones.items()}}
        cur_cost = self.total_time(cur)

        for benefit, c, st in cand_list:
            if benefit <= 1e-9:
                break
            new_truck, new_drones = self._apply_convert_truck_to_drone(cur["truck"], cur["drones"], c, st)
            if new_truck is None:
                continue
            if not self._is_solution_feasible(new_truck, new_drones):
                continue
            cand = {"truck": new_truck, "drones": new_drones}
            cand_cost = self.total_time(cand)
            if cand_cost < cur_cost:
                cur, cur_cost = cand, cand_cost
                loc = rebuild_loc()

        return cur["truck"], cur["drones"]

    def _best_station_roundtrip_time(self, client: str):
        best_st, best_time = None, float("inf")
        for s in self.stations:
            d = self._dist(s, client) + self._dist(client, s)
            if math.isinf(d):
                continue
            t = d / self.v_drone
            if t < best_time:
                best_time, best_st = t, s
        return best_st, best_time

    def _best_station_insertion_delta_time(self, route: List[str], st: str):
        best = float("inf")
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            delta = (self._dist(u, st) + self._dist(st, v) - self._dist(u, v)) / self.v_truck
            best = min(best, delta)
        return best

    def _apply_convert_truck_to_drone(self, truck_routes, drones, client: str, st: str):
        truck = [r[:] for r in truck_routes]
        dr = {s: [rt[:] for rt in routes] for s, routes in drones.items()}
        dr.setdefault(st, [])
        if self.demand[client] > self.drone_capacity:
            return None, None
        if len(dr[st]) >= self.drone_count.get(st, 1):
            return None, None

        found = None
        for ridx, r in enumerate(truck):
            if client in r:
                found = (ridx, r.index(client))
                break
        if found is None:
            return None, None

        ridx, pos = found
        r = truck[ridx][:]
        r.pop(pos)

        if st not in r:
            best_pos, best_delta = 1, float("inf")
            for i in range(len(r) - 1):
                u, v = r[i], r[i + 1]
                delta = self._dist(u, st) + self._dist(st, v) - self._dist(u, v)
                if delta < best_delta:
                    best_delta, best_pos = delta, i + 1
            r.insert(best_pos, st)

        truck[ridx] = r
        dr[st].append([st, client, st])
        return truck, dr

    def _is_solution_feasible(self, truck_routes, drone_routes):
        cset = set(self.clients)
        served = {c: 0 for c in cset}

        # truck
        for r in truck_routes:
            load = 0.0
            for n in r:
                if n in cset:
                    load += self.demand[n]
                    if load > self.Q:
                        return False
                    served[n] += 1

        # drones
        for st, routes in drone_routes.items():
            if routes and not any(st in r for r in truck_routes):
                return False
            if len(routes) > self.drone_count.get(st, 1):
                return False
            for rt in routes:
                if len(rt) != 3:
                    return False
                a, c, b = rt
                if a != st or b != st:
                    return False
                if c not in served:
                    return False
                served[c] += 1

        return all(served[c] == 1 for c in served)

    # --- kmeans, xy, infer ---
    def _kmeans_customers(self, customers, K, iters=50):
        K = max(1, min(K, len(customers)))
        init = self.rng.sample(customers, K)
        centroids = [self.node_xy[c] for c in init]

        for _ in range(iters):
            clusters = [[] for _ in range(K)]
            for c in customers:
                x, y = self.node_xy[c]
                best = min(range(K), key=lambda k: (x - centroids[k][0])**2 + (y - centroids[k][1])**2)
                clusters[best].append(c)

            new_centroids = []
            for k in range(K):
                if not clusters[k]:
                    pick = self.rng.choice(customers)
                    new_centroids.append(self.node_xy[pick])
                else:
                    xs = [self.node_xy[c][0] for c in clusters[k]]
                    ys = [self.node_xy[c][1] for c in clusters[k]]
                    new_centroids.append((sum(xs)/len(xs), sum(ys)/len(ys)))
            if new_centroids == centroids:
                break
            centroids = new_centroids

        return clusters, centroids

    def _build_xy(self, final_data):
        node_xy = {}
        if isinstance(final_data, dict):
            for node, rec in final_data.items():
                node_xy[node] = (float(rec[2]), float(rec[3]))
        else:
            for rec in final_data:
                node_xy[rec[0]] = (float(rec[2]), float(rec[3]))
        return node_xy

    def _infer_truck_count(self):
        C = self.p.get("C", None)
        if isinstance(C, int) and C >= 1:
            return C
        total_dem = sum(self.demand[c] for c in self.clients)
        return max(1, math.ceil(total_dem / self.Q))

    def _dist(self, u, v):
        d = self.arcs.get((u, v))
        if d is None:
            d = self.arcs.get((u.replace("_end",""), v.replace("_end","")), float("inf"))
        return d if d is not None else float("inf")


# -----------------------------
# GVNS (paper-style k=1..4 shaking procedures)
# -----------------------------
class GVNS_PaperStyle:
    def __init__(self, parameters: Dict[str, Any], seed: int = 0,
                 v_truck: float = 1.0, v_drone: float = 3.0,
                 drone_capacity: float = 50.0,
                 drone_count: Optional[Dict[str, int]] = None):

        self.p = parameters
        self.rng = random.Random(seed)

        self.clients = set(parameters["clients"])
        self.stations = set(parameters["stations"])
        self.depot = parameters["depot_start"][0]
        self.depot_end = parameters["depot_end"][0] if isinstance(parameters["depot_end"], list) else parameters["depot_end"]

        self.arcs = parameters["arcs"]
        self.demand = parameters["demand"]
        self.Q = parameters["Q"]

        self.v_truck = float(v_truck)
        self.v_drone = float(v_drone)

        self.drone_capacity = float(drone_capacity)
        self.drone_count = drone_count if drone_count is not None else {s: 3 for s in self.stations}

        # Paper parameters
        self.k_max = 4
        self.time_limit = 3000.0
        self.use_nmax = True  # set False if you only want wall-clock comparison
        self.n_max = None     # computed from n and k_max

        # Build paper CH
        self.ch = PaperCHForDroneStations(parameters, seed=seed,
                                          v_truck=v_truck, v_drone=v_drone,
                                          drone_capacity=drone_capacity,
                                          drone_count=self.drone_count)

    # ---------- objective ----------
    def total_time(self, sol: Dict[str, Any]) -> float:
        total = 0.0
        for r in sol.get("truck", []):
            for i in range(len(r)-1):
                total += self._dist(r[i], r[i+1]) / self.v_truck
        for st, routes in sol.get("drones", {}).items():
            for rt in routes:
                for i in range(len(rt)-1):
                    total += self._dist(rt[i], rt[i+1]) / self.v_drone
        return total

    # ---------- feasibility (same semantics as your current checker) ----------
    def is_solution_feasible(self, truck_routes, drone_routes) -> bool:
        served = {c: 0 for c in self.clients}

        for r in truck_routes:
            load = 0.0
            for n in r:
                if n in self.clients:
                    load += self.demand[n]
                    if load > self.Q:
                        return False
                    served[n] += 1

        for st, routes in drone_routes.items():
            if routes and not any(st in r for r in truck_routes):
                return False
            if len(routes) > self.drone_count.get(st, 1):
                return False
            for rt in routes:
                if len(rt) != 3:
                    return False
                a, c, b = rt
                if a != st or b != st:
                    return False
                if c not in served:
                    return False
                if self.demand[c] > self.drone_capacity:
                    return False
                served[c] += 1

        return all(served[c] == 1 for c in served)

    # ---------- solve ----------
    def solve(self) -> Dict[str, Any]:
        # --- initial solution from paper CH ---
        S = self.ch.build()
        S["cost"] = self.total_time(S)
        best = copy.deepcopy(S)

        # --- paper parameters ---
        n = len(self.clients)  # number of customers
        k_max = self.k_max  # paper uses kmax=4 in experiments
        n_max = int((n * n) / k_max)  # nmax = n^2 / kmax  (paper)
        deltaT = self.time_limit  # δT (paper uses 1h, you can set it)

        # --- paper cont logic ---
        cont = 0  # cont ← 0
        t0 = time.time()

        # stop if elapsed >= δT OR cont >= nmax
        while (time.time() - t0) < deltaT and cont < n_max:
            improvement = False  # improvement ← False  (per outer iteration)

            k = 1
            while k <= k_max and (time.time() - t0) < deltaT:
                S_prime = self.shake_k(S, k)  # shaking with neighborhood index k
                S_pp = self.rvnd(S_prime)  # RVND local search

                if S_pp["cost"] < S["cost"]:
                    S = S_pp
                    improvement = True
                    k = 1  # reset k to 1 on improvement
                    if S["cost"] < best["cost"]:
                        best = copy.deepcopy(S)
                else:
                    k += 1  # try next k

            # after exploring k=1..kmax, update cont exactly once (paper)
            if improvement:
                cont = 0
            else:
                cont += 1

        return best

    # ---------- paper-style shaking procedures ----------
    def shake_k(self, S: Dict[str, Any], k: int) -> Dict[str, Any]:
        # "first feasible random neighbor" style: try a bounded number of random attempts
        tries = 60
        for _ in range(tries):
            cand = None
            if k == 1:
                # Relocate to Drone (largest-ish in original paper ordering can differ,
                # but we keep 4 fixed procedures; order is chosen by increasing structural impact.)
                cand = self._shake_relocate_to_drone(S)
            elif k == 2:
                # Relocate to Truck
                cand = self._shake_relocate_to_truck(S)
            elif k == 3:
                # Exchange
                cand = self._shake_exchange(S)
            elif k == 4:
                # Relocate to Station-drone (replacement of "Relocate to Robot")
                cand = self._shake_relocate_to_station_drone(S)

            if cand is not None:
                cand["cost"] = self.total_time(cand)
                return cand

        # if no feasible neighbor found, return original (paper would just move on)
        out = copy.deepcopy(S)
        out["cost"] = self.total_time(out)
        return out

    # ----- shaking operator 1: Relocate to Drone (truck->drone) -----
    def _shake_relocate_to_drone(self, S):
        truck = [r[:] for r in S["truck"]]
        drones = {st: [rt[:] for rt in routes] for st, routes in S.get("drones", {}).items()}

        # pick a truck-served customer
        candidates = []
        for ridx, r in enumerate(truck):
            for i, n in enumerate(r):
                if n in self.clients:
                    candidates.append((ridx, i, n))
        if not candidates:
            return None

        ridx, i, c = self.rng.choice(candidates)
        if self.demand[c] > self.drone_capacity:
            return None

        st = self._select_best_station_for_client(c)
        if st is None:
            return None

        drones.setdefault(st, [])
        if len(drones[st]) >= self.drone_count.get(st, 1):
            return None

        # remove c from its route
        r = truck[ridx][:]
        r.pop(i)

        # ensure station visited (insert if needed)
        if st not in r:
            best_pos, best_delta = 1, float("inf")
            for p in range(len(r)-1):
                u, v = r[p], r[p+1]
                delta = self._dist(u, st) + self._dist(st, v) - self._dist(u, v)
                if delta < best_delta:
                    best_delta, best_pos = delta, p+1
            r.insert(best_pos, st)

        truck[ridx] = r
        drones[st].append([st, c, st])

        cand = {"truck": truck, "drones": {s: rs for s, rs in drones.items() if rs}}
        if not self.is_solution_feasible(cand["truck"], cand["drones"]):
            return None
        return cand

    # ----- shaking operator 2: Relocate to Truck (drone->truck) -----
    def _shake_relocate_to_truck(self, S):
        truck = [r[:] for r in S["truck"]]
        drones = {st: [rt[:] for rt in routes] for st, routes in S.get("drones", {}).items()}

        # pick a drone-served customer
        drone_tasks = []
        for st, routes in drones.items():
            for idx, rt in enumerate(routes):
                if len(rt) == 3:
                    drone_tasks.append((st, idx, rt[1]))
        if not drone_tasks:
            return None

        st, idx, c = self.rng.choice(drone_tasks)

        # remove drone task
        drones[st] = [rt for j, rt in enumerate(drones[st]) if j != idx]

        # insert c into a random route at a random position
        ridx = self.rng.randrange(len(truck))
        r = truck[ridx][:]
        pos = self.rng.randrange(1, len(r))  # allow before depot_end
        r.insert(pos, c)

        # capacity check
        if not self._route_capacity_ok(r):
            return None
        truck[ridx] = r

        cand = {"truck": truck, "drones": {s: rs for s, rs in drones.items() if rs}}
        if not self.is_solution_feasible(cand["truck"], cand["drones"]):
            return None
        return cand

    # ----- shaking operator 3: Exchange (swap two truck-served customers) -----
    def _shake_exchange(self, S):
        truck = [r[:] for r in S["truck"]]
        drones = {st: [rt[:] for rt in routes] for st, routes in S.get("drones", {}).items()}

        positions = []
        for ridx, r in enumerate(truck):
            for i, n in enumerate(r):
                if n in self.clients:
                    positions.append((ridx, i))
        if len(positions) < 2:
            return None

        (r1, i1), (r2, i2) = self.rng.sample(positions, 2)
        truck[r1][i1], truck[r2][i2] = truck[r2][i2], truck[r1][i1]

        if not self._route_capacity_ok(truck[r1]) or not self._route_capacity_ok(truck[r2]):
            return None

        cand = {"truck": truck, "drones": {s: rs for s, rs in drones.items() if rs}}
        if not self.is_solution_feasible(cand["truck"], cand["drones"]):
            return None
        return cand

    # ----- shaking operator 4: Relocate to Station-drone (replace "Relocate to Robot") -----
    # Here: pick a truck-served customer and force it to be served by a station drone, but
    # also try a RANDOM station (not best one) to enlarge neighborhood.
    def _shake_relocate_to_station_drone(self, S):
        truck = [r[:] for r in S["truck"]]
        drones = {st: [rt[:] for rt in routes] for st, routes in S.get("drones", {}).items()}

        candidates = []
        for ridx, r in enumerate(truck):
            for i, n in enumerate(r):
                if n in self.clients:
                    candidates.append((ridx, i, n))
        if not candidates:
            return None

        ridx, i, c = self.rng.choice(candidates)
        if self.demand[c] > self.drone_capacity:
            return None

        st = self.rng.choice(list(self.stations))  # random station increases neighborhood size
        drones.setdefault(st, [])
        if len(drones[st]) >= self.drone_count.get(st, 1):
            return None

        r = truck[ridx][:]
        r.pop(i)

        if st not in r:
            best_pos, best_delta = 1, float("inf")
            for p in range(len(r)-1):
                u, v = r[p], r[p+1]
                delta = self._dist(u, st) + self._dist(st, v) - self._dist(u, v)
                if delta < best_delta:
                    best_delta, best_pos = delta, p+1
            r.insert(best_pos, st)

        truck[ridx] = r
        drones[st].append([st, c, st])

        cand = {"truck": truck, "drones": {s: rs for s, rs in drones.items() if rs}}
        if not self.is_solution_feasible(cand["truck"], cand["drones"]):
            return None
        return cand

    # ---------- RVND (kept simple for runtime predictability) ----------
    # We use first-improvement sampling rather than full enumeration to avoid slowdowns.
    def rvnd(self, S: Dict[str, Any]) -> Dict[str, Any]:
        neighborhoods = [
            self._ls_first_improve_relocate_within_route,
            self._ls_first_improve_exchange,
            self._ls_first_improve_drone_toggle,
        ]
        N = neighborhoods[:]
        self.rng.shuffle(N)

        cur = copy.deepcopy(S)
        cur["cost"] = self.total_time(cur)

        while N:
            nbh = self.rng.choice(N)
            improved = nbh(cur)
            if improved is not None and improved["cost"] < cur["cost"]:
                cur = improved
                N = neighborhoods[:]
                self.rng.shuffle(N)
            else:
                N.remove(nbh)

        return cur

    def _ls_first_improve_relocate_within_route(self, S):
        # sample a few random relocations within a route, accept first improvement
        trials = 80
        base = S["cost"]
        for _ in range(trials):
            truck = [r[:] for r in S["truck"]]
            drones = S.get("drones", {})
            ridx = self.rng.randrange(len(truck))
            r = truck[ridx]
            cust_pos = [i for i, n in enumerate(r) if n in self.clients]
            if len(cust_pos) < 2:
                continue
            i = self.rng.choice(cust_pos)
            j = self.rng.randrange(1, len(r)-1)
            if i == j:
                continue
            new_r = r[:]
            node = new_r.pop(i)
            new_r.insert(j, node)
            if not self._route_capacity_ok(new_r):
                continue
            truck[ridx] = new_r
            cand = {"truck": truck, "drones": drones}
            if not self.is_solution_feasible(cand["truck"], cand["drones"]):
                continue
            cand["cost"] = self.total_time(cand)
            if cand["cost"] < base:
                return cand
        return None

    def _ls_first_improve_exchange(self, S):
        trials = 80
        base = S["cost"]
        for _ in range(trials):
            truck = [r[:] for r in S["truck"]]
            drones = S.get("drones", {})
            pos = []
            for ridx, r in enumerate(truck):
                for i, n in enumerate(r):
                    if n in self.clients:
                        pos.append((ridx, i))
            if len(pos) < 2:
                return None
            (r1, i1), (r2, i2) = self.rng.sample(pos, 2)
            truck[r1][i1], truck[r2][i2] = truck[r2][i2], truck[r1][i1]
            if not self._route_capacity_ok(truck[r1]) or not self._route_capacity_ok(truck[r2]):
                continue
            cand = {"truck": truck, "drones": drones}
            if not self.is_solution_feasible(cand["truck"], cand["drones"]):
                continue
            cand["cost"] = self.total_time(cand)
            if cand["cost"] < base:
                return cand
        return None

    def _ls_first_improve_drone_toggle(self, S):
        trials = 60
        base = S["cost"]
        for _ in range(trials):
            c = self.rng.choice(list(self.clients))
            cand = self._try_toggle_client(S, c)
            if cand is None:
                continue
            cand["cost"] = self.total_time(cand)
            if cand["cost"] < base:
                return cand
        return None

    def _try_toggle_client(self, S, client):
        # if client drone-served -> insert into truck
        drones = {st: [rt[:] for rt in routes] for st, routes in S.get("drones", {}).items()}
        truck = [r[:] for r in S["truck"]]

        served_station = None
        served_idx = None
        for st, routes in drones.items():
            for idx, rt in enumerate(routes):
                if len(rt) == 3 and rt[1] == client:
                    served_station, served_idx = st, idx
                    break
            if served_station is not None:
                break

        if served_station is not None:
            # drone->truck
            drones[served_station] = [rt for j, rt in enumerate(drones[served_station]) if j != served_idx]
            ridx = self.rng.randrange(len(truck))
            r = truck[ridx][:]
            pos = self.rng.randrange(1, len(r))
            r.insert(pos, client)
            if not self._route_capacity_ok(r):
                return None
            truck[ridx] = r
            cand = {"truck": truck, "drones": {s: rs for s, rs in drones.items() if rs}}
            if not self.is_solution_feasible(cand["truck"], cand["drones"]):
                return None
            return cand
        else:
            # truck->drone
            if self.demand[client] > self.drone_capacity:
                return None
            st = self._select_best_station_for_client(client)
            if st is None:
                return None
            drones.setdefault(st, [])
            if len(drones[st]) >= self.drone_count.get(st, 1):
                return None
            # find client in truck
            found = None
            for ridx, r in enumerate(truck):
                if client in r:
                    found = (ridx, r.index(client))
                    break
            if found is None:
                return None
            ridx, i = found
            r = truck[ridx][:]
            r.pop(i)
            if st not in r:
                best_pos, best_delta = 1, float("inf")
                for p in range(len(r)-1):
                    u, v = r[p], r[p+1]
                    delta = self._dist(u, st) + self._dist(st, v) - self._dist(u, v)
                    if delta < best_delta:
                        best_delta, best_pos = delta, p+1
                r.insert(best_pos, st)
            truck[ridx] = r
            drones[st].append([st, client, st])
            cand = {"truck": truck, "drones": {s: rs for s, rs in drones.items() if rs}}
            if not self.is_solution_feasible(cand["truck"], cand["drones"]):
                return None
            return cand

    # ---------- helpers ----------
    def _route_capacity_ok(self, route):
        load = 0.0
        for n in route:
            if n in self.clients:
                load += self.demand[n]
                if load > self.Q:
                    return False
        return True

    def _select_best_station_for_client(self, client):
        best_station, best_t = None, float("inf")
        for s in self.stations:
            d = self._dist(s, client) + self._dist(client, s)
            if math.isinf(d):
                continue
            t = d / self.v_drone
            if t < best_t:
                best_t, best_station = t, s
        return best_station

    def _dist(self, u, v):
        d = self.arcs.get((u, v))
        if d is None:
            d = self.arcs.get((u.replace("_end",""), v.replace("_end","")), float("inf"))
        return d if d is not None else float("inf")



