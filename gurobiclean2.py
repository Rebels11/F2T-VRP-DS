import math
import gurobipy as gp
from gurobipy import GRB
from file_reader import get_parameters


def solve_vrpd_milp_station_once_fast(
    file_path: str,
    v_truck: float = 1.0,
    v_drone: float = 2.0,
    drones_per_station: int = 3,
    time_limit: int = 3600,
    threads: int = 0,
    mip_gap: float | None = None,
    k_buffer: int = 3,
    k_max: int = 8,
):
    """
    Fast MILP variant:
    - Tight truck upper bound Kmax (instead of K=|clients|)
    - Remove SCF flow variables; enforce connectivity by Lazy subtour cuts per truck
    - Keep: station visited at most once globally + truck/drone service exactly once
    """

    params = get_parameters(file_path, 0)

    clients = list(params["clients"])
    stations = list(params["stations"])
    depot = params["depot_start"][0]

    demand = params["demand"]
    Q = params["Q"]
    arcs = params["arcs"]

    # -----------------------------
    # Nodes & arcs
    # -----------------------------
    N = [depot] + clients + stations

    A = []
    for i in N:
        for j in N:
            if i != j and (i, j) in arcs:
                A.append((i, j))

    # -----------------------------
    # (1) Tight vehicle upper bound
    # -----------------------------
    total_demand = sum(float(demand[c]) for c in clients)
    lb_trucks = int(math.ceil(total_demand / float(Q))) if Q > 0 else 1
    if k_max is None:
        Kmax = min(len(clients), lb_trucks + int(k_buffer))
    else:
        Kmax = min(len(clients), int(k_max))
    K = list(range(Kmax))

    m = gp.Model("VRPD_station_once_fast")
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit
    if threads and threads > 0:
        m.Params.Threads = threads
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap

    # (2) Enable lazy constraints for subtour elimination
    m.Params.LazyConstraints = 1

    # -------- Variables --------
    use = m.addVars(K, vtype=GRB.BINARY, name="use")
    x = m.addVars(A, K, vtype=GRB.BINARY, name="x")
    y = m.addVars(clients, K, vtype=GRB.BINARY, name="y")
    yT = m.addVars(clients, vtype=GRB.BINARY, name="yT")
    w = m.addVars(stations, clients, vtype=GRB.BINARY, name="w")
    visitS = m.addVars(stations, vtype=GRB.BINARY, name="visitS")

    # -------- Objective (time) --------
    truck_time = gp.quicksum(arcs[i, j] * x[i, j, k] for (i, j) in A for k in K) / v_truck
    drone_time = gp.quicksum((2.0 * arcs[s, c]) * w[s, c] for s in stations for c in clients) / v_drone
    m.setObjective(truck_time + drone_time, GRB.MINIMIZE)

    # -------- Constraints --------

    # (1) Each customer served exactly once: truck OR drone
    for c in clients:
        m.addConstr(yT[c] + gp.quicksum(w[s, c] for s in stations) == 1, name=f"serve_once[{c}]")

    # (2) Link yT and y[c,k]
    for c in clients:
        m.addConstr(gp.quicksum(y[c, k] for k in K) == yT[c], name=f"truck_assign[{c}]")

    # (3) Drone limit per station
    for s in stations:
        m.addConstr(gp.quicksum(w[s, c] for c in clients) <= drones_per_station, name=f"drone_cap[{s}]")

    # (4) Depot constraints per truck: if used, leave depot once and return once
    for k in K:
        out_depot = gp.quicksum(x[depot, j, k] for j in N if j != depot and (depot, j) in arcs)
        in_depot = gp.quicksum(x[i, depot, k] for i in N if i != depot and (i, depot) in arcs)
        m.addConstr(out_depot == use[k], name=f"depot_out[{k}]")
        m.addConstr(in_depot == use[k], name=f"depot_in[{k}]")

        # (optional but usually helpful): avoid empty truck doing depot->station->depot
        m.addConstr(gp.quicksum(y[c, k] for c in clients) >= use[k], name=f"nonempty[{k}]")

    # (5) Customer degree constraints per assigned truck
    for c in clients:
        for k in K:
            in_ck = gp.quicksum(x[i, c, k] for i in N if i != c and (i, c) in arcs)
            out_ck = gp.quicksum(x[c, j, k] for j in N if j != c and (c, j) in arcs)
            m.addConstr(in_ck == y[c, k], name=f"cust_in[{c},{k}]")
            m.addConstr(out_ck == y[c, k], name=f"cust_out[{c},{k}]")

    # (6) Station flow per truck: in == out
    # + bind station usage to use[k] (important to prevent station-only subtours for unused trucks)
    for s in stations:
        for k in K:
            in_sk = gp.quicksum(x[i, s, k] for i in N if i != s and (i, s) in arcs)
            out_sk = gp.quicksum(x[s, j, k] for j in N if j != s and (s, j) in arcs)
            m.addConstr(in_sk == out_sk, name=f"station_flow[{s},{k}]")
            m.addConstr(in_sk <= use[k], name=f"station_only_if_use[{s},{k}]")

    # (7) Station visited at most once globally
    for s in stations:
        incoming_all = gp.quicksum(x[i, s, k] for k in K for i in N if i != s and (i, s) in arcs)
        outgoing_all = gp.quicksum(x[s, j, k] for k in K for j in N if j != s and (s, j) in arcs)
        m.addConstr(incoming_all == visitS[s], name=f"visit_in[{s}]")
        m.addConstr(outgoing_all == visitS[s], name=f"visit_out[{s}]")

        for c in clients:
            m.addConstr(w[s, c] <= visitS[s], name=f"drone_only_if_visit[{s},{c}]")

    # (8) Capacity per truck
    for k in K:
        m.addConstr(
            gp.quicksum(demand[c] * y[c, k] for c in clients) <= Q * use[k],
            name=f"cap[{k}]"
        )

    # -----------------------------
    # (9) Lazy subtour elimination callback (per truck)
    # -----------------------------
    def _extract_cycles(selected_arcs, nodes):
        """
        selected_arcs: list of (i,j) with x(i,j)=1 (for a fixed k)
        return list of cycles (each cycle as list of nodes), excluding those that contain depot
        """
        out_map = {i: j for i, j in selected_arcs}  # because degrees are 1-in-1-out on customers, this is safe-ish
        visited = set()
        cycles = []
        for start in nodes:
            if start in visited or start not in out_map:
                continue
            cur = start
            path = []
            seen_local = {}
            while True:
                if cur in seen_local:
                    # found a cycle
                    cycle_start_idx = seen_local[cur]
                    cycle = path[cycle_start_idx:]
                    cycles.append(cycle)
                    break
                if cur in visited:
                    break
                seen_local[cur] = len(path)
                visited.add(cur)
                path.append(cur)
                nxt = out_map.get(cur)
                if nxt is None:
                    break
                cur = nxt
        return cycles

    def subtour_cb(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        # for each truck k, find a subtour that does not include depot, then add a cut
        for k in K:
            if model.cbGetSolution(use[k]) < 0.5:
                continue

            sel = []
            for (i, j) in A:
                if model.cbGetSolution(x[i, j, k]) > 0.5:
                    sel.append((i, j))

            if not sel:
                continue

            # nodes that appear in this truck's route
            nodes_in_route = set()
            for i, j in sel:
                nodes_in_route.add(i)
                nodes_in_route.add(j)

            if depot not in nodes_in_route:
                # should not happen due to depot_out/depot_in, but safe guard
                continue

            # find cycles; cut any cycle not containing depot
            cycles = _extract_cycles(sel, list(nodes_in_route))
            for cyc in cycles:
                if depot in cyc:
                    continue
                # subtour cut: sum_{i in S, j in S} x[i,j,k] <= |S|-1
                S = set(cyc)
                expr = gp.quicksum(x[i, j, k] for i in S for j in S if i != j and (i, j) in arcs)
                model.cbLazy(expr <= len(S) - 1)

    # -------- Solve --------

    m.Params.MIPFocus = 1  #
    m.Params.Heuristics = 0.8  #
    m.Params.RINS = 20  #
    m.Params.Presolve = 2  #
    m.Params.Cuts = 1  #
    m.optimize(subtour_cb)
    status = m.Status
    runtime = m.Runtime

    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        return {"status": status, "runtime": runtime, "objective": None, "truck_routes": [], "drone_tasks": []}

    obj = m.ObjVal

    # -------- Extract solution --------
    drone_tasks = [(s, c) for s in stations for c in clients if w[s, c].X > 0.5]

    truck_routes = []
    for k in K:
        if use[k].X < 0.5:
            continue
        next_of = {}
        for (i, j) in A:
            if x[i, j, k].X > 0.5:
                next_of[i] = j

        if depot not in next_of:
            continue

        route = [depot]
        cur = depot
        guard = 0
        while True:
            nxt = next_of.get(cur)
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt
            if cur == depot:
                break
            guard += 1
            if guard > 100000:
                break
        truck_routes.append(route)

    return {"status": status, "runtime": runtime, "objective": obj, "truck_routes": truck_routes, "drone_tasks": drone_tasks}


