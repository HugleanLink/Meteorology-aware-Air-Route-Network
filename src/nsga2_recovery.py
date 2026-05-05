from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

_PROJECT_ROOT = Path(__file__).parent.parent


def _load_config():
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Candidate generation ─────────────────────────────────────────────

def build_recovery_candidates(network, damage_result):
    """
    Generate recovery candidate set from damage assessment result.

    Candidates:
      - reactivate affected edges (cost=0.3)
      - reactivate failed edges (cost=1.0)
      - detour for each failed edge (cost=1.5, F_norm=0.7*original)

    Returns list of candidate dicts.
    """
    trunk_edges = network["trunk_edges"].copy()
    trunk_edge_status = damage_result["trunk_edge_status"]
    branch_edge_status = damage_result.get("branch_edge_status")
    candidates = []

    # ── Trunk reactivate / detour candidates ──
    for _, row in trunk_edge_status.iterrows():
        status = row["damage_status"]
        if status == "affected":
            candidates.append({
                "edge_id": row["edge_id"],
                "type": "reactivate",
                "cost": 0.3,
                "center_i": int(row["center_i"]),
                "center_j": int(row["center_j"]),
                "F_norm_ij": float(trunk_edges.loc[
                    trunk_edges["edge_id"] == row["edge_id"], "F_norm_ij"
                ].values[0]),
            })
        elif status == "failed":
            candidates.append({
                "edge_id": row["edge_id"],
                "type": "reactivate",
                "cost": 1.0,
                "center_i": int(row["center_i"]),
                "center_j": int(row["center_j"]),
                "F_norm_ij": float(trunk_edges.loc[
                    trunk_edges["edge_id"] == row["edge_id"], "F_norm_ij"
                ].values[0]),
            })
            # Detour candidate for each failed edge
            candidates.append({
                "edge_id": f"detour_{row['edge_id']}",
                "type": "detour",
                "cost": 1.5,
                "center_i": int(row["center_i"]),
                "center_j": int(row["center_j"]),
                "F_norm_ij": float(trunk_edges.loc[
                    trunk_edges["edge_id"] == row["edge_id"], "F_norm_ij"
                ].values[0]) * 0.7,
            })

    # ── Branch reactivate candidates (only for failed branches causing isolation) ──
    if branch_edge_status is not None and len(branch_edge_status) > 0:
        for _, row in branch_edge_status.iterrows():
            if row["damage_status"] == "failed":
                candidates.append({
                    "edge_id": row["edge_id"],
                    "type": "branch_reactivate",
                    "cost": 0.5,
                    "center_i": -1,  # placeholder, branch not in trunk graph
                    "center_j": -1,
                    "F_norm_ij": 0.0,
                    "poi_id": row["poi_id"],
                })

    n_trunk = sum(1 for c in candidates if c["type"] in ("reactivate", "detour"))
    n_branch = sum(1 for c in candidates if c["type"] == "branch_reactivate")
    print(f"恢复候选方案: {len(candidates)} 个 "
          f"(主干: {n_trunk}, 支线: {n_branch})")
    for c in candidates:
        if c["type"] == "branch_reactivate":
            print(f"  [branch] {c['edge_id']}  cost={c['cost']:.1f}")
        else:
            print(f"  [{c['type']:>10}] {c['edge_id']:>12}  "
                  f"({c['center_i']}-{c['center_j']})  "
                  f"cost={c['cost']:.1f}  F_norm={c['F_norm_ij']:.4f}")

    return candidates


# ── Recovery data helpers ────────────────────────────────────────────

def _compute_base_state(damage_result, trunk_edges):
    """Extract base normal trunk edges and branch state from damage result."""
    trunk_edge_status = damage_result["trunk_edge_status"]
    normal_edges = set(
        trunk_edge_status.loc[trunk_edge_status["damage_status"] == "normal", "edge_id"]
    )
    F_all = trunk_edges["F_norm_ij"].sum()
    F_normal = trunk_edges.loc[
        trunk_edges["edge_id"].isin(normal_edges), "F_norm_ij"
    ].sum()

    n_isolated_base = damage_result.get("n_isolated_demand", 0)
    n_total_branch = len(damage_result.get("branch_edge_status", []))
    n_total_branch = n_total_branch if n_total_branch > 0 else 1

    return normal_edges, F_normal, F_all, n_isolated_base, n_total_branch


def _evaluate_solution(x_mask, candidates, base_normal_edges, F_normal_base, F_all,
                       n_total_trunk, n_isolated_base, n_total_branch):
    """Evaluate a binary selection vector. Returns (R_struct, isolation_rate, total_cost, ...)."""
    activated_trunk = set()
    activated_branch = set()
    total_cost = 0.0
    F_add = 0.0

    for i, selected in enumerate(x_mask):
        if selected:
            c = candidates[i]
            total_cost += c["cost"]
            if c["type"] == "branch_reactivate":
                activated_branch.add(c["edge_id"])
            elif c["type"] == "detour":
                activated_trunk.add(c["edge_id"])
                F_add += c["F_norm_ij"]
            else:  # reactivate
                activated_trunk.add(c["edge_id"])

    # For reactivate edges whose original id has no corresponding detour
    for i, selected in enumerate(x_mask):
        if selected and candidates[i]["type"] == "reactivate":
            cid = candidates[i]["edge_id"]
            detour_id = f"detour_{cid}"
            if detour_id not in activated_trunk:
                F_add += candidates[i]["F_norm_ij"]

    # Count new normal trunk edges
    original_ids = set()
    for eid in activated_trunk:
        original_ids.add(eid.replace("detour_", ""))
    n_trunk_recovered = len(original_ids - base_normal_edges)

    n_trunk_normal = len(base_normal_edges) + n_trunk_recovered
    R_struct = n_trunk_normal / n_total_trunk

    # Branch isolation rate after recovery
    n_isolated_after = max(0, n_isolated_base - len(activated_branch))
    isolation_rate = n_isolated_after / n_total_branch if n_total_branch > 0 else 0.0

    return R_struct, isolation_rate, total_cost, n_trunk_normal, n_isolated_after


# ── pymoo Problem ────────────────────────────────────────────────────

class RecoveryProblem(Problem):
    """
    Binary optimization for network recovery.
    Minimize: f1 = 1 - R_struct,  f2 = isolation_rate
    Subject to: sum(cost * x) <= budget
    """

    def __init__(self, candidates, base_normal_edges, F_normal_base, F_all,
                 n_total_trunk, n_isolated_base, n_total_branch, budget):
        self.candidates = candidates
        self.base_normal_edges = base_normal_edges
        self.F_normal_base = F_normal_base
        self.F_all = F_all
        self.n_total_trunk = n_total_trunk
        self.n_isolated_base = n_isolated_base
        self.n_total_branch = n_total_branch
        self.budget = budget

        n_vars = len(candidates)
        super().__init__(
            n_var=n_vars,
            n_obj=2,
            n_ieq_constr=1,
            xl=0,
            xu=1,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = len(X)
        F = np.zeros((n_pop, 2))
        G = np.zeros((n_pop, 1))

        for i in range(n_pop):
            x_mask = np.asarray(X[i], dtype=bool)

            R_struct, isolation_rate, total_cost, _, _ = _evaluate_solution(
                x_mask, self.candidates, self.base_normal_edges,
                self.F_normal_base, self.F_all,
                self.n_total_trunk, self.n_isolated_base, self.n_total_branch
            )

            F[i, 0] = 1.0 - R_struct
            F[i, 1] = isolation_rate
            G[i, 0] = total_cost - self.budget

        out["F"] = F
        out["G"] = G


# ── NSGA-II runner ───────────────────────────────────────────────────

def run_nsga2(problem, config):
    """
    Run NSGA-II optimization and return pymoo Result.
    """
    cfg = config.get("nsga2", {})
    pop_size = cfg.get("pop_size", 80)
    n_gen = cfg.get("n_gen", 200)
    n_vars = len(problem.candidates)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=BinomialCrossover(prob=0.9),
        mutation=BitflipMutation(prob=1.0 / n_vars if n_vars > 0 else 0.05),
        eliminate_duplicates=True,
    )

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-4,
        cvtol=1e-4,
        ftol=1e-4,
        period=30,
        n_max_gen=n_gen,
    )

    print(f"\nNSGA-II 优化: pop_size={pop_size}, n_gen={n_gen}, "
          f"n_vars={n_vars}, budget={problem.budget:.1f}")

    result = minimize(problem, algorithm, termination, seed=config["project"].get("random_state", 42),
                      verbose=False)

    print(f"优化完成: {len(result.pop)} 个解决方案")
    print(f"  Pareto 前沿: {len(result.opt)} 个")

    return result


# ── Pareto extraction ────────────────────────────────────────────────

def extract_pareto_solutions(result, candidates, damage_result, trunk_edges):
    """
    Extract Pareto front solutions into a DataFrame.
    """
    base_normal_edges, F_normal_base, F_all, n_isolated_base, n_total_branch = \
        _compute_base_state(damage_result, trunk_edges)
    n_total_trunk = len(trunk_edges)

    if result.opt is not None:
        opt_X = result.opt.get("X")
        opt_F = result.opt.get("F")
    else:
        opt_X = result.pop.get("X")
        opt_F = result.pop.get("F")

    solutions = []
    for sol_id, (x_row, f_row) in enumerate(zip(opt_X, opt_F)):
        x_mask = np.array(x_row, dtype=bool)

        R_struct, isolation_rate, total_cost, _, n_isolated_after = _evaluate_solution(
            x_mask, candidates, base_normal_edges, F_normal_base, F_all,
            n_total_trunk, n_isolated_base, n_total_branch
        )
        R_combined = 0.5 * R_struct + 0.5 * (1.0 - isolation_rate)

        selected_edges = []
        n_react = 0
        n_det = 0
        n_branch_rec = 0
        for i, sel in enumerate(x_mask):
            if sel:
                selected_edges.append(candidates[i]["edge_id"])
                ct = candidates[i]["type"]
                if ct == "branch_reactivate":
                    n_branch_rec += 1
                elif ct == "detour":
                    n_det += 1
                else:
                    n_react += 1

        solutions.append({
            "solution_id": sol_id,
            "R_struct": R_struct,
            "isolation_rate": isolation_rate,
            "R_combined": R_combined,
            "total_cost": total_cost,
            "n_reactivated": n_react,
            "n_detour": n_det,
            "n_branch_recovered": n_branch_rec,
            "n_isolated_after": n_isolated_after,
            "selected_edges": ",".join(selected_edges) if len(selected_edges) <= 20 else
                ",".join(selected_edges[:20]) + f"...(+{len(selected_edges)-20})",
        })

    pareto_df = pd.DataFrame(solutions)
    pareto_df = pareto_df.sort_values(
        ["R_combined"], ascending=False
    ).reset_index(drop=True)

    print(f"\nPareto 前沿解: {len(pareto_df)} 个")
    cols = ["solution_id", "R_struct", "isolation_rate", "R_combined",
            "total_cost", "n_reactivated", "n_detour", "n_branch_recovered"]
    print(pareto_df[cols].to_string(index=False))

    return pareto_df


# ── Recommendation ───────────────────────────────────────────────────

def recommend_recovery(pareto_df):
    """
    Recommend a balanced solution from the Pareto front.
    Chooses the solution closest to ideal point (R_struct=1, isolation_rate=0).
    """
    if len(pareto_df) == 0:
        raise ValueError("Pareto 前沿为空，无法推荐方案")

    ideal = np.array([1.0, 0.0])
    dists = np.sqrt(
        (pareto_df["R_struct"].values - ideal[0]) ** 2
        + (pareto_df["isolation_rate"].values - ideal[1]) ** 2
    )
    best_idx = np.argmin(dists)

    recommended = pareto_df.iloc[best_idx].to_dict()

    print(f"\n推荐恢复方案 (solution_id={recommended['solution_id']}):")
    print(f"  R_struct      = {recommended['R_struct']:.4f}")
    print(f"  isolation_rate = {recommended['isolation_rate']:.4f}")
    print(f"  R_combined    = {recommended['R_combined']:.4f}")
    print(f"  总成本        = {recommended['total_cost']:.2f}")
    print(f"  主干重新激活  = {recommended['n_reactivated']} 条")
    print(f"  绕行边        = {recommended['n_detour']} 条")
    print(f"  支线恢复      = {recommended['n_branch_recovered']} 条")
    if recommended["selected_edges"]:
        print(f"  选中边: {recommended['selected_edges']}")

    return recommended


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.damage_assessment import load_network, assess_damage

    config = _load_config()

    print("=" * 60)
    print("阶段4-任务2: NSGA-II 航网恢复优化 (含支线)")
    print("=" * 60)

    network = load_network()
    pool = "风雨复合"

    print(f"\n{'─'*50}")
    print(f"评估气象池: {pool}")
    damage = assess_damage(network, pool, config)
    print(f"  R_struct = {damage['R_struct']:.4f}  "
          f"R_flow = {damage['R_flow']:.4f}  "
          f"R_combined = {damage['R_combined']:.4f}")
    print(f"  主干失效: {damage['n_trunk_failed']}  "
          f"支线失效: {damage['n_branch_failed']}  "
          f"孤立需求: {damage['n_isolated_demand']}")

    if damage["R_struct"] < 1.0 or damage["n_isolated_demand"] > 0:
        candidates = build_recovery_candidates(network, damage)

        trunk_edges = network["trunk_edges"]
        base_normal_edges, F_normal_base, F_all, n_isolated_base, n_total_branch = \
            _compute_base_state(damage, trunk_edges)
        n_total_trunk = len(trunk_edges)

        avg_trunk_cost = np.mean([c["cost"] for c in candidates if c["type"] != "branch_reactivate"] or [1.0])
        avg_branch_cost = 0.5
        n_trunk_cand = sum(1 for c in candidates if c["type"] != "branch_reactivate")
        n_branch_cand = sum(1 for c in candidates if c["type"] == "branch_reactivate")
        budget = (n_trunk_cand * avg_trunk_cost + n_branch_cand * avg_branch_cost) * 0.5

        problem = RecoveryProblem(
            candidates, base_normal_edges, F_normal_base, F_all,
            n_total_trunk, n_isolated_base, n_total_branch, budget
        )
        result = run_nsga2(problem, config)

        pareto_df = extract_pareto_solutions(result, candidates, damage, trunk_edges)
        recommended = recommend_recovery(pareto_df)

        output_dir = _PROJECT_ROOT / "outputs" / "recovery"
        output_dir.mkdir(parents=True, exist_ok=True)
        pareto_df.to_csv(output_dir / f"pareto_solutions_{pool}.csv", index=False)
        print(f"\nPareto 解保存至: outputs/recovery/pareto_solutions_{pool}.csv")
    else:
        print(f"该气象池下航网无需恢复")
