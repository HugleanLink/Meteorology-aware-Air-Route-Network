from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from coordinate import haversine_distance


def compute_gravity_flow(centers: pd.DataFrame, config: dict) -> pd.DataFrame:
    beta = config["gravity_flow"]["beta"]
    n = len(centers)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            ci = centers.iloc[i]
            cj = centers.iloc[j]
            d_ij = haversine_distance(ci["lon"], ci["lat"], cj["lon"], cj["lat"])
            f_hat = ci["P_i"] * cj["P_i"] / (d_ij ** beta)
            rows.append({
                "center_i": ci["center_id"],
                "center_j": cj["center_id"],
                "d_ij": d_ij,
                "F_hat_ij": f_hat,
            })

    df = pd.DataFrame(rows, columns=["center_i", "center_j", "d_ij", "F_hat_ij"])

    f_min = df["F_hat_ij"].min()
    f_max = df["F_hat_ij"].max()
    if f_max == f_min:
        df["F_norm_ij"] = 1.0
    else:
        df["F_norm_ij"] = (df["F_hat_ij"] - f_min) / (f_max - f_min)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    centers = pd.read_csv(
        project_root / "outputs" / "network" / "main_centers.csv", encoding="utf-8-sig"
    )

    df_flow = compute_gravity_flow(centers, config)

    print(f"Total edges: {len(df_flow)}")
    print(f"F_norm_ij max: {df_flow['F_norm_ij'].max():.6f}")
    print(f"F_norm_ij min: {df_flow['F_norm_ij'].min():.6f}")

    output_path = project_root / "outputs" / "network" / "gravity_flow.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_flow.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")
