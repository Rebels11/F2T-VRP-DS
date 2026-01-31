import gurobipy as gp
from gurobipy import GRB
from file_reader import get_parameters


def solve_vrpd_milp_station_once(
    file_path: str,
    v_truck: float = 1.0,
    v_drone: float = 2.0,
    drones_per_station: int = 3,
    time_limit: int = 3600,
    threads: int = 0,
    mip_gap: float = None,
):

    params = get_parameters(file_path, 0)

    clients = list(params["clients"])
    stations = list(params["stations"])
    depot = params["depot_start"][0]

    demand = params["demand"]
    Q = params["Q"]
    arcs = params["arcs"]

    # Nodes
    N = [depot] + clients + stations

    # Directed arc set where distance exists
    A = []
    for i in N:
        for j in N:
            if i != j and (i, j) in arcs:
                A.append((i, j))

    # Max number of trucks (safe upper bound)
    K = list(range(len(clients)))

    m = gp.Model("VRPD_station_once")
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit
    if threads and threads > 0:
        m.Params.Threads = threads
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap

    # -------- Variables --------
    use = m.addVars(K, vtype=GRB.BINARY, name="use")

    # x[i,j,k] binary arc usage for truck k
    x = m.addVars(A, K, vtype=GRB.BINARY, name="x")

    # y[c,k] = 1 if customer c served by truck k
    y = m.addVars(clients, K, vtype=GRB.BINARY, name="y")

    # yT[c] = 1 if customer c served by truck (any k)
    yT = m.addVars(clients, vtype=GRB.BINARY, name="yT")

    # w[s,c] = 1 if customer c served by drone from station s
    w = m.addVars(stations, clients, vtype=GRB.BINARY, name="w")

    # visitS[s] = 1 if station s is visited by any truck
    visitS = m.addVars(stations, vtype=GRB.BINARY, name="visitS")

    # SCF flow for subtour elimination & capacity consistency
    f = m.addVars(A, K, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="f")

    # -------- Objective (time) --------
    truck_time = gp.quicksum(arcs[i, j] * x[i, j, k] for (i, j) in A for k in K) / v_truck

    # arcs symmetric => round-trip distance = 2 * arcs[s,c]
    drone_time = gp.quicksum((2.0 * arcs[s, c]) * w[s, c] for s in stations for c in clients) / v_drone

    m.setObjective(truck_time + drone_time, GRB.MINIMIZE)

    # -------- Constraints --------

    # (1) Each customer served exactly once: truck OR drone
    for c in clients:
        m.addConstr(
            yT[c] + gp.quicksum(w[s, c] for s in stations) == 1,
            name=f"serve_once[{c}]"
        )

    # (2) Link yT and y[c,k]
    for c in clients:
        m.addConstr(
            gp.quicksum(y[c, k] for k in K) == yT[c],
            name=f"truck_assign[{c}]"
        )

    # (3) Drone limit per station
    for s in stations:
        m.addConstr(
            gp.quicksum(w[s, c] for c in clients) <= drones_per_station,
            name=f"drone_cap[{s}]"
        )

    # (4) Depot constraints per truck: if used, leave depot once and return once
    for k in K:
        out_depot = gp.quicksum(x[depot, j, k] for j in N if j != depot and (depot, j) in arcs)
        in_depot = gp.quicksum(x[i, depot, k] for i in N if i != depot and (i, depot) in arcs)
        m.addConstr(out_depot == use[k], name=f"depot_out[{k}]")
        m.addConstr(in_depot == use[k], name=f"depot_in[{k}]")

    # (5) Customer degree constraints per assigned truck:
    # if y[c,k]=1 then exactly one in and one out on truck k, else 0.
    for c in clients:
        for k in K:
            in_ck = gp.quicksum(x[i, c, k] for i in N if i != c and (i, c) in arcs)
            out_ck = gp.quicksum(x[c, j, k] for j in N if j != c and (c, j) in arcs)
            m.addConstr(in_ck == y[c, k], name=f"cust_in[{c},{k}]")
            m.addConstr(out_ck == y[c, k], name=f"cust_out[{c},{k}]")

    # (6) Station flow per truck: in == out (visit or not visit), but NO degree upper bound here.
    # The "visited at most once globally" will be enforced separately.
    for s in stations:
        for k in K:
            in_sk = gp.quicksum(x[i, s, k] for i in N if i != s and (i, s) in arcs)
            out_sk = gp.quicksum(x[s, j, k] for j in N if j != s and (s, j) in arcs)
            m.addConstr(in_sk == out_sk, name=f"station_flow[{s},{k}]")

    # (7) Station visited at most once globally:
    # Sum of incoming arcs over all trucks equals visitS[s], so it can be 0 or 1.
    for s in stations:
        incoming_all = gp.quicksum(x[i, s, k] for k in K for i in N if i != s and (i, s) in arcs)
        outgoing_all = gp.quicksum(x[s, j, k] for k in K for j in N if j != s and (s, j) in arcs)

        # If visited, must have exactly one incoming and one outgoing in the whole solution
        m.addConstr(incoming_all == visitS[s], name=f"visit_in[{s}]")
        m.addConstr(outgoing_all == visitS[s], name=f"visit_out[{s}]")

        # Drone only if station visited
        for c in clients:
            m.addConstr(w[s, c] <= visitS[s], name=f"drone_only_if_visit[{s},{c}]")

    # (8) Capacity per truck: sum demand of customers served by truck k <= Q
    for k in K:
        m.addConstr(
            gp.quicksum(demand[c] * y[c, k] for c in clients) <= Q * use[k],
            name=f"cap[{k}]"
        )

    # (9) Subtour elimination / connectivity via single-commodity flow (SCF)
    # Flow can only go on used arcs
    for (i, j) in A:
        for k in K:
            m.addConstr(f[i, j, k] <= Q * x[i, j, k], name=f"flow_cap[{i},{j},{k}]")

    for k in K:
        total_demand_k = gp.quicksum(demand[c] * y[c, k] for c in clients)

        out_flow_depot = gp.quicksum(f[depot, j, k] for j in N if j != depot and (depot, j) in arcs)
        in_flow_depot = gp.quicksum(f[i, depot, k] for i in N if i != depot and (i, depot) in arcs)
        m.addConstr(out_flow_depot - in_flow_depot == total_demand_k, name=f"depot_flow[{k}]")

        for c in clients:
            inflow = gp.quicksum(f[i, c, k] for i in N if i != c and (i, c) in arcs)
            outflow = gp.quicksum(f[c, j, k] for j in N if j != c and (c, j) in arcs)
            m.addConstr(inflow - outflow == demand[c] * y[c, k], name=f"cust_flow[{c},{k}]")

        for s in stations:
            inflow = gp.quicksum(f[i, s, k] for i in N if i != s and (i, s) in arcs)
            outflow = gp.quicksum(f[s, j, k] for j in N if j != s and (s, j) in arcs)
            m.addConstr(inflow - outflow == 0, name=f"station_flow_bal[{s},{k}]")

    # (10) symmetry breaking: use[k] non-increasing
    for k in range(len(K) - 1):
        m.addConstr(use[k] >= use[k + 1], name=f"sym_use[{k}]")

    # -------- Solve --------
    m.optimize()

    status = m.Status
    runtime = m.Runtime

    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        return {
            "status": status,
            "runtime": runtime,
            "objective": None,
            "truck_routes": [],
            "drone_tasks": [],
        }

    obj = m.ObjVal

    # -------- Extract solution --------
    drone_tasks = [(s, c) for s in stations for c in clients if w[s, c].X > 0.5]

    truck_routes = []
    for k in K:
        if use[k].X < 0.5:
            continue

        # Build adjacency mapping for this truck (now degrees at stations are <=1 globally,
        # and customers are exactly 1 in/out if served => route is a simple cycle from depot to depot)
        next_of = {}
        for (i, j) in A:
            if x[i, j, k].X > 0.5:
                next_of[i] = j

        if depot not in next_of:
            continue

        # Follow from depot until return to depot
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
            if guard > 10000:
                break

        truck_routes.append(route)

    return {
        "status": status,
        "runtime": runtime,
        "objective": obj,
        "truck_routes": truck_routes,
        "drone_tasks": drone_tasks,
    }

