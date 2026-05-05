import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from coordinate import haversine_distance, build_nofly_polygons, edge_intersects_nofly


def generate_candidate_edges(centers: pd.DataFrame, config: dict) -> tuple:
    k_default = config["trunk_network"]["k_edge_default"]
    k_max = config["trunk_network"]["k_edge_max"]
    n = len(centers)

    lons = centers["lon"].values
    lats = centers["lat"].values
    nofly_polygons, _ = build_nofly_polygons(config)

    def _build_edges(k_val):
        rows = []
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                d = haversine_distance(lons[i], lats[i], lons[j], lats[j])
                dists.append((j, d))
            dists.sort(key=lambda x: x[1])
            for j, d in dists[:k_val]:
                if not edge_intersects_nofly(lons[i], lats[i], lons[j], lats[j], nofly_polygons):
                    rows.append({"center_i": min(i, j), "center_j": max(i, j), "d_ij": d})
        df = pd.DataFrame(rows).drop_duplicates(subset=["center_i", "center_j"]).reset_index(drop=True)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for _, row in df.iterrows():
            G.add_edge(int(row["center_i"]), int(row["center_j"]))
        return df, G

    k_used = k_default
    for k in range(k_default, k_max + 1):
        edges, G = _build_edges(k)
        if nx.is_connected(G):
            k_used = k
            break
    else:
        edges, G = _build_edges(k_max)
        k_used = k_max

    if not nx.is_connected(G):
        print(f"Warning: graph still not connected at k_edge={k_max}")

    # Count how many raw KNN edges were filtered by nofly
    total_pairs = n * (n - 1) // 2
    print(f"禁飞区过滤后候选边: {len(edges)} 条 (k_edge={k_used})")

    d_all = edges["d_ij"]
    d_max = d_all.max()
    d_min = d_all.min()
    edges["D_norm_ij"] = 1.0 if d_max == d_min else (d_all - d_min) / (d_max - d_min)

    edges["F_norm_ij"] = np.nan
    edges["Cost_ij"] = np.nan

    return edges[["center_i", "center_j", "d_ij", "D_norm_ij", "F_norm_ij", "Cost_ij"]], k_used


def build_trunk_network(centers: pd.DataFrame, candidate_edges: pd.DataFrame, config: dict) -> dict:
    aco_cfg = config["aco"]
    trunk_cfg = config["trunk_network"]
    n_ants = aco_cfg["n_ants"]
    n_iter = aco_cfg["n_iter"]
    alpha = aco_cfg["alpha"]
    beta = aco_cfg["beta_aco"]
    rho = aco_cfg["rho"]
    q = aco_cfg["q"]
    epsilon = aco_cfg["epsilon"]
    redundant_ratio = trunk_cfg["redundant_edge_ratio"]

    n = len(centers)
    edge_list = list(zip(
        candidate_edges["center_i"].astype(int),
        candidate_edges["center_j"].astype(int),
        candidate_edges["Cost_ij"],
    ))
    num_edges = len(edge_list)
    edge_idx_map = {(int(ci), int(cj)): idx for idx, (ci, cj, _) in enumerate(edge_list)}

    tau = np.ones(num_edges)
    eta = np.array([1.0 / (cost + epsilon) for _, _, cost in edge_list])

    best_cost_global = float("inf")
    best_tree_global = None
    rng = np.random.RandomState(config["project"]["random_state"])

    for iteration in range(n_iter):
        ant_trees = []
        ant_costs = []

        for _ in range(n_ants):
            visited = {rng.randint(0, n)}
            tree_edges = []
            total_cost = 0.0

            while len(visited) < n:
                Omega = []
                for idx, (ci, cj, _) in enumerate(edge_list):
                    if (ci in visited) != (cj in visited):
                        Omega.append(idx)

                if not Omega:
                    unvisited = set(range(n)) - visited
                    for u in unvisited:
                        for v in visited:
                            key = (min(u, v), max(u, v))
                            if key in edge_idx_map:
                                Omega.append(edge_idx_map[key])
                    if not Omega:
                        break

                probs = np.array([tau[idx] ** alpha * eta[idx] ** beta for idx in Omega])
                probs = probs / probs.sum()
                chosen_idx = rng.choice(Omega, p=probs)
                ci, cj, cost = edge_list[chosen_idx]

                tree_edges.append((ci, cj))
                total_cost += cost
                visited.add(ci)
                visited.add(cj)

            if len(visited) == n:
                ant_trees.append(tree_edges)
                ant_costs.append(total_cost)

        if not ant_trees:
            continue

        best_idx = int(np.argmin(ant_costs))
        if ant_costs[best_idx] < best_cost_global:
            best_cost_global = ant_costs[best_idx]
            best_tree_global = ant_trees[best_idx]

        tau = tau * (1 - rho)
        for ci, cj in ant_trees[best_idx]:
            key = (ci, cj)
            if key in edge_idx_map:
                tau[edge_idx_map[key]] += q / ant_costs[best_idx]

    E_base = best_tree_global if best_tree_global else []

    base_set = set(E_base)
    nofly_polygons, _ = build_nofly_polygons(config)
    non_base = [
        (ci, cj, cost) for ci, cj, cost in [
            (e[0], e[1], e[2]) for e in edge_list
        ]
        if (ci, cj) not in base_set
        and not edge_intersects_nofly(
            centers.iloc[ci]["lon"], centers.iloc[ci]["lat"],
            centers.iloc[cj]["lon"], centers.iloc[cj]["lat"],
            nofly_polygons)
    ]
    non_base.sort(key=lambda x: x[2])
    r = math.ceil(redundant_ratio * n)
    E_red = [(ci, cj) for ci, cj, _ in non_base[:r]]

    E_trunk_records = []
    for ci, cj in E_base:
        idx = edge_idx_map[(ci, cj)]
        row = candidate_edges.iloc[idx]
        _, _, cost = edge_list[idx]
        E_trunk_records.append({
            "center_i": ci, "center_j": cj,
            "d_ij": row["d_ij"], "F_norm_ij": row["F_norm_ij"],
            "Cost_ij": cost, "is_redundant": False,
        })
    for ci, cj in E_red:
        idx = edge_idx_map[(ci, cj)]
        row = candidate_edges.iloc[idx]
        _, _, cost = edge_list[idx]
        E_trunk_records.append({
            "center_i": ci, "center_j": cj,
            "d_ij": row["d_ij"], "F_norm_ij": row["F_norm_ij"],
            "Cost_ij": cost, "is_redundant": True,
        })

    E_trunk = pd.DataFrame(E_trunk_records, columns=[
        "center_i", "center_j", "d_ij", "F_norm_ij", "Cost_ij", "is_redundant",
    ])

    return {
        "E_base": E_base,
        "E_red": E_red,
        "E_trunk": E_trunk,
        "best_cost": best_cost_global,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    centers = pd.read_csv(project_root / "outputs" / "network" / "main_centers.csv")
    gravity = pd.read_csv(project_root / "outputs" / "network" / "gravity_flow.csv")

    edges, k_used = generate_candidate_edges(centers, config)

    f_map = {}
    for _, row in gravity.iterrows():
        f_map[(int(row["center_i"]), int(row["center_j"]))] = row["F_norm_ij"]
    for idx, row in edges.iterrows():
        key = (int(row["center_i"]), int(row["center_j"]))
        edges.at[idx, "F_norm_ij"] = f_map.get(key, f_map.get((key[1], key[0]), 0.0))

    cost_w_d = config["trunk_network"]["cost_weight_distance"]
    cost_w_f = config["trunk_network"]["cost_weight_flow"]
    edges["Cost_ij"] = cost_w_d * edges["D_norm_ij"] + cost_w_f * (1 - edges["F_norm_ij"])

    print(f"Candidate edges: {len(edges)}")
    print(f"k_edge used for connectivity: {k_used}")

    result = build_trunk_network(centers, edges, config)

    print(f"ACO best tree cost: {result['best_cost']:.6f}")
    print(f"Base edges: {len(result['E_base'])}")
    print(f"Redundant edges: {len(result['E_red'])}")
    print(f"Total trunk edges: {len(result['E_trunk'])}")

    output_path = project_root / "outputs" / "network" / "trunk_edges.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["E_trunk"].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")
