from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def _load_config():
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_network():
    """
    Load all network data files and return a unified network dict.

    Returns dict with keys: centers, trunk_edges, branch_edges, G
    """
    network_dir = _PROJECT_ROOT / "outputs" / "network"
    pred_path = _PROJECT_ROOT / "data" / "processed" / "meteo_edge_predictions.parquet"

    centers = pd.read_csv(network_dir / "main_centers.csv")
    trunk_edges = pd.read_csv(network_dir / "trunk_edges.csv")
    branch_edges = pd.read_csv(network_dir / "branch_edges.csv")
    predictions = pd.read_parquet(pred_path)

    # Aggregate pred_status by edge_id using mode
    mode_status = predictions.groupby("edge_id")["pred_status"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    trunk_edges["edge_id"] = trunk_edges.apply(
        lambda r: f"{int(r['center_i'])}-{int(r['center_j'])}", axis=1
    )
    trunk_edges["pred_status"] = trunk_edges["edge_id"].map(mode_status)

    # Build trunk graph
    G = nx.Graph()
    for _, row in trunk_edges.iterrows():
        G.add_edge(
            int(row["center_i"]), int(row["center_j"]),
            edge_id=row["edge_id"],
            F_norm_ij=row["F_norm_ij"],
            is_redundant=row["is_redundant"],
        )

    print(f"网络加载完成:")
    print(f"  主中心: {len(centers)}")
    n_base = (~trunk_edges["is_redundant"].astype(bool)).sum()
    n_red = trunk_edges["is_redundant"].astype(bool).sum()
    print(f"  主干边: {len(trunk_edges)} (基干 {n_base} + 冗余 {n_red})")
    print(f"  支线: {len(branch_edges)}")
    print(f"  图节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")

    return {
        "centers": centers,
        "trunk_edges": trunk_edges,
        "branch_edges": branch_edges,
        "G": G,
    }


def _assess_edge_set(edges_df, pool_data, fail_threshold, affected_threshold):
    """Assess damage for a set of edges (trunk or branch) and return status per edge."""
    fail_probs = []
    affected_probs = []
    statuses = []

    for _, edge in edges_df.iterrows():
        eid = edge["edge_id"]
        edge_data = pool_data[pool_data["edge_id"] == eid]
        n_steps = len(edge_data)

        if n_steps == 0:
            fail_probs.append(0.0)
            affected_probs.append(0.0)
            statuses.append("normal")
            continue

        fail_count = (edge_data["pred_status"] == 2).sum()
        aff_count = (edge_data["pred_status"] == 1).sum()

        f_prob = fail_count / n_steps
        a_prob = aff_count / n_steps

        fail_probs.append(f_prob)
        affected_probs.append(a_prob)

        if f_prob >= fail_threshold:
            statuses.append("failed")
        elif a_prob >= affected_threshold:
            statuses.append("affected")
        else:
            statuses.append("normal")

    result = edges_df.copy()
    result["fail_prob"] = fail_probs
    result["affected_prob"] = affected_probs
    result["damage_status"] = statuses
    return result


def assess_damage(network, weather_pool, config):
    """
    Assess network damage under a specific weather pool.

    Evaluates trunk edges (R_struct, R_flow) and branch edges (isolation).

    Returns dict with:
      R_struct, R_flow, R_combined,
      branch_fail_rate, n_isolated_demand,
      trunk_edge_status, branch_edge_status
    """
    pred_path = _PROJECT_ROOT / "data" / "processed" / "meteo_edge_predictions_all.parquet"
    if not pred_path.exists():
        pred_path = _PROJECT_ROOT / "data" / "processed" / "meteo_edge_predictions.parquet"
    predictions = pd.read_parquet(pred_path)

    pool_data = predictions[predictions["weather_pool"] == weather_pool]
    if len(pool_data) == 0:
        raise ValueError(f"找不到气象池 '{weather_pool}' 的预测数据")

    cfg = config.get("damage_assessment", {})
    fail_threshold = cfg.get("fail_prob_threshold", 0.25)
    affected_threshold = cfg.get("affected_prob_threshold", 0.10)

    # ── Trunk assessment ──
    trunk_edges = network["trunk_edges"].copy()
    trunk_result = _assess_edge_set(trunk_edges, pool_data, fail_threshold, affected_threshold)

    trunk_statuses = trunk_result["damage_status"].tolist()
    n_trunk_normal = sum(s == "normal" for s in trunk_statuses)
    n_trunk_failed = sum(s == "failed" for s in trunk_statuses)
    n_trunk_affected = sum(s == "affected" for s in trunk_statuses)
    n_trunk_total = len(trunk_edges)

    R_struct = n_trunk_normal / n_trunk_total

    normal_mask = np.array(trunk_statuses) == "normal"
    F_all = trunk_edges["F_norm_ij"].values.sum()
    F_normal = trunk_edges.loc[normal_mask, "F_norm_ij"].sum()
    R_flow = F_normal / F_all if F_all > 0 else 0.0

    # Trunk connectivity
    G_normal = nx.Graph()
    for _, row in trunk_result.iterrows():
        if row["damage_status"] == "normal":
            G_normal.add_edge(int(row["center_i"]), int(row["center_j"]))
    for node in network["G"].nodes():
        if node not in G_normal:
            G_normal.add_node(node)
    connected = nx.is_connected(G_normal) if G_normal.number_of_nodes() > 0 else False

    # ── Branch assessment ──
    branch_edges = network["branch_edges"].copy()
    # Build edge_id for branches: "B_{poi_id}"
    branch_edges["edge_id"] = "B_" + branch_edges["poi_id"].astype(str)

    branch_result = _assess_edge_set(branch_edges, pool_data, fail_threshold, affected_threshold)

    n_branch_total = len(branch_edges)
    n_branch_failed = sum(branch_result["damage_status"] == "failed")
    n_branch_affected = sum(branch_result["damage_status"] == "affected")
    branch_fail_rate = n_branch_failed / n_branch_total if n_branch_total > 0 else 0.0

    # Isolation: only failed branches cause isolation
    n_isolated = n_branch_failed
    total_demand = n_branch_total  # each branch serves one core demand point
    isolation_rate = n_isolated / total_demand if total_demand > 0 else 0.0

    R_combined = 0.5 * R_struct + 0.5 * (1.0 - isolation_rate)

    # ── Build outputs ──
    trunk_status_df = trunk_result[["center_i", "center_j", "edge_id",
                                     "fail_prob", "affected_prob", "damage_status"]]
    branch_status_df = branch_result[["poi_id", "edge_id",
                                       "fail_prob", "affected_prob", "damage_status"]]

    return {
        "weather_pool": weather_pool,
        "R_struct": R_struct,
        "R_flow": R_flow,
        "R_combined": R_combined,
        "branch_fail_rate": branch_fail_rate,
        "n_isolated_demand": n_isolated,
        "isolation_rate": isolation_rate,
        "connected": connected,
        "n_trunk_failed": n_trunk_failed,
        "n_trunk_affected": n_trunk_affected,
        "n_branch_failed": n_branch_failed,
        "n_branch_affected": n_branch_affected,
        "trunk_edge_status": trunk_status_df,
        "branch_edge_status": branch_status_df,
    }


if __name__ == "__main__":
    config = _load_config()

    print("=" * 60)
    print("阶段4-任务1: 受损航网评估 (含支线)")
    print("=" * 60)

    network = load_network()

    for pool in ["极端风", "极端雨", "风雨复合"]:
        print(f"\n{'─'*50}")
        damage = assess_damage(network, pool, config)
        print(f"气象池: {pool}")
        print(f"  R_struct = {damage['R_struct']:.4f}  "
              f"R_flow = {damage['R_flow']:.4f}  "
              f"R_combined = {damage['R_combined']:.4f}")
        print(f"  主干: 失效 {damage['n_trunk_failed']}, "
              f"受影响 {damage['n_trunk_affected']}")
        print(f"  支线: 失效 {damage['n_branch_failed']}, "
              f"受影响 {damage['n_branch_affected']}, "
              f"fail_rate = {damage['branch_fail_rate']:.4f}")
        print(f"  孤立需求点: {damage['n_isolated_demand']} / "
              f"{len(network['branch_edges'])}")
        print(f"  连通性: {'是' if damage['connected'] else '否'}")
        # Print trunk edge status summary
        print(f"  主干边明细:")
        for _, row in damage["trunk_edge_status"].iterrows():
            status_icon = {"normal": "N", "affected": "A", "failed": "F"}
            icon = status_icon.get(row["damage_status"], "?")
            print(f"    [{icon}] {row['edge_id']}: fail={row['fail_prob']:.3f} "
                  f"aff={row['affected_prob']:.3f} -> {row['damage_status']}")
        # Branch summary only
        n_f = damage["n_branch_failed"]
        n_a = damage["n_branch_affected"]
        n_n = len(network["branch_edges"]) - n_f - n_a
        print(f"  支线汇总: [N]={n_n}, [A]={n_a}, [F]={n_f}")
