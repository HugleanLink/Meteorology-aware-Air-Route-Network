from pathlib import Path
import sys

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.damage_assessment import load_network, assess_damage
from src.nsga2_recovery import (
    build_recovery_candidates,
    RecoveryProblem,
    run_nsga2,
    extract_pareto_solutions,
    recommend_recovery,
    _compute_base_state,
)


def _load_config():
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch():
    config = _load_config()
    pools = ["极端风", "极端雨", "风雨复合"]

    print("=" * 60)
    print("阶段4: 批量气象池损伤评估与恢复优化")
    print("=" * 60)

    network = load_network()
    trunk_edges = network["trunk_edges"]
    n_total_edges = len(trunk_edges)

    output_dir = _PROJECT_ROOT / "outputs" / "recovery"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for pool in pools:
        print(f"\n{'='*60}")
        print(f"气象池: {pool}")
        print(f"{'='*60}")

        damage = assess_damage(network, pool, config)

        if damage["R_struct"] < 1.0 or damage["n_isolated_demand"] > 0:
            candidates = build_recovery_candidates(network, damage)

            base_normal_edges, F_normal_base, F_all, n_isolated_base, n_total_branch = \
                _compute_base_state(damage, trunk_edges)

            n_trunk_cand = sum(1 for c in candidates if c["type"] != "branch_reactivate")
            n_branch_cand = sum(1 for c in candidates if c["type"] == "branch_reactivate")
            avg_trunk = np.mean([c["cost"] for c in candidates if c["type"] != "branch_reactivate"] or [1.0])
            budget = (n_trunk_cand * avg_trunk + n_branch_cand * 0.5) * 0.5

            problem = RecoveryProblem(
                candidates, base_normal_edges, F_normal_base, F_all,
                n_total_edges, n_isolated_base, n_total_branch, budget
            )
            result = run_nsga2(problem, config)

            pareto_df = extract_pareto_solutions(result, candidates, damage, trunk_edges)
            recommended = recommend_recovery(pareto_df)

            pareto_path = output_dir / f"pareto_solutions_{pool}.csv"
            pareto_df.to_csv(pareto_path, index=False)
            print(f"\nPareto 解保存至: {pareto_path}")

            summary_rows.append({
                "weather_pool": pool,
                "R_struct_before": round(damage["R_struct"], 4),
                "R_flow_before": round(damage["R_flow"], 4),
                "branch_fail_rate": round(damage["branch_fail_rate"], 4),
                "n_isolated": damage["n_isolated_demand"],
                "R_combined_before": round(damage["R_combined"], 4),
                "R_combined_after": round(recommended["R_combined"], 4),
                "n_pareto": len(pareto_df),
            })
        else:
            print(f"  该气象池下航网无需恢复")
            summary_rows.append({
                "weather_pool": pool,
                "R_struct_before": round(damage["R_struct"], 4),
                "R_flow_before": round(damage["R_flow"], 4),
                "branch_fail_rate": round(damage["branch_fail_rate"], 4),
                "n_isolated": damage["n_isolated_demand"],
                "R_combined_before": round(damage["R_combined"], 4),
                "R_combined_after": round(damage["R_combined"], 4),
                "n_pareto": 0,
            })

    # Summary table
    print(f"\n{'='*90}")
    print("批量模拟汇总 (含支线)")
    print(f"{'='*90}")
    print(f"{'weather_pool':<12} {'R_struct':>9} {'R_flow':>9} {'br_fail':>8} "
          f"{'n_isol':>7} {'R_comb_pre':>11} {'R_comb_post':>12} {'n_pareto':>9}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['weather_pool']:<12} {row['R_struct_before']:>9.4f} "
              f"{row['R_flow_before']:>9.4f} {row['branch_fail_rate']:>8.4f} "
              f"{row['n_isolated']:>7} {row['R_combined_before']:>11.4f} "
              f"{row['R_combined_after']:>12.4f} {row['n_pareto']:>9}")


if __name__ == "__main__":
    run_batch()
