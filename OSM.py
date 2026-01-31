import random
import math
import networkx as nx
import osmnx as ox

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_tokyo_small_parameters_compatible(
    center=(35.7155, 139.7770), dist=1800,
    n_customers=40, n_stations=6,
    seed=20251224,
    v_truck=1.0, v_drone=3.0,
):
    import random, math
    import networkx as nx
    import osmnx as ox

    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
        return 2*R*math.asin(math.sqrt(a))

    random.seed(seed)

    G = ox.graph_from_point(center, dist=dist, network_type="drive", simplify=True)

    nodes = list(G.nodes)
    chosen = random.sample(nodes, k=(1 + n_customers + n_stations))
    depot_node = chosen[0]
    cust_nodes = chosen[1:1+n_customers]
    st_nodes   = chosen[1+n_customers:]

    depot_start_id = 0
    clients = list(range(1, 1+n_customers))  # 1..40
    stations = list(range(1+n_customers, 1+n_customers+n_stations))  # 41..46
    depot_end_id = 1 + n_customers + n_stations  # 47
    all_nodes = [depot_start_id] + clients + stations + [depot_end_id]

    id2node = {depot_start_id: depot_node, depot_end_id: depot_node}
    id2node.update({i: n for i, n in zip(clients, cust_nodes)})
    id2node.update({i: n for i, n in zip(stations, st_nodes)})

    final_data = {i: (G.nodes[id2node[i]]["x"], G.nodes[id2node[i]]["y"]) for i in all_nodes}  # (lon,lat)


    truck_dist_m = {}
    truck_time = {}
    for i in all_nodes:
        si = id2node[i]
        lengths = nx.single_source_dijkstra_path_length(G, si, weight="length")
        for j in all_nodes:
            if i == j:
                continue
            tj = id2node[j]
            d = float(lengths.get(tj, float("inf")))
            truck_dist_m[(i, j)] = d
            truck_time[(i, j)] = d / float(v_truck)

    # ✅ add self-loops
    for i in all_nodes:
        truck_dist_m[(i, i)] = 0.0
        truck_time[(i, i)] = 0.0

    # --- drone straight-line time
    drone_time = {}
    for i in all_nodes:
        lon1, lat1 = final_data[i]
        for j in all_nodes:
            if i == j:
                continue
            lon2, lat2 = final_data[j]
            d_m = haversine_m(lat1, lon1, lat2, lon2)
            drone_time[(i, j)] = d_m / float(v_drone)
        # ✅ add self-loops
    for i in all_nodes:
        drone_time[(i, i)] = 0.0

    # demands etc.
    demand = {i: 0 for i in all_nodes}
    for c in clients:
        demand[c] = random.randint(1, 10)

    ready_time = {i: 0 for i in all_nodes}
    due_date = {i: 10**9 for i in all_nodes}
    service_time = {i: 0 for i in all_nodes}

    Q = 200
    C = 200
    g = 0.0
    h = 0.0
    v = v_truck  #


    depot_start = [depot_start_id]
    depot_end = [depot_end_id]

    #
    arcs = truck_time


    times = truck_time

    #
    arc_list = list(truck_time.keys())

    parameters = {
        "clients": clients,
        "stations": stations,
        "all_nodes": all_nodes,
        "depot_start": depot_start,
        "depot_end": depot_end,

        "demand": demand,
        "ready_time": ready_time,
        "due_date": due_date,
        "service_time": service_time,

        "arcs": arcs,         # dict cost, compatible with your Heuristic
        "times": times,       # dict cost
        "arc_list": arc_list, # optional edge list

        "final_data": final_data,
        "original_stations": stations.copy(),

        "Q": Q,
        "C": C,
        "g": g,
        "h": h,
        "v": v,

        # extras for later plotting/debug
        "_G_osm": G,
        "_id2node": id2node,
        "_truck_dist_m": truck_dist_m,
        "_truck_time": truck_time,
        "_drone_time": drone_time,
    }
    return parameters
