from pathlib import Path

import pandas as pd
import yaml


def clean_poi(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    valid_types = set(config["poi"]["core_types"] + config["poi"]["auxiliary_types"])
    core_types = set(config["poi"]["core_types"])
    weights = config["poi"]["weights"]

    df = df.dropna(subset=["lon", "lat"])

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])

    df = df[df["type"].isin(valid_types)]

    df["is_core"] = df["type"].isin(core_types)
    df["weight"] = df["type"].map(weights)

    df = df.reset_index(drop=True)

    return df[["poi_id", "name", "type", "lon", "lat", "district", "is_core", "weight"]]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_path = project_root / "outputs" / "poi" / "poi_raw.csv"
    df_raw = pd.read_csv(input_path, encoding="utf-8-sig")

    df_clean = clean_poi(df_raw, config)

    print(f"清洗前总条数: {len(df_raw)}")
    print(f"清洗后总条数: {len(df_clean)}")
    print(f"核心需求点数量: {df_clean['is_core'].sum()}")
    print(f"\n各区核心需求点数量:")
    core_by_district = df_clean[df_clean["is_core"]].groupby("district").size()
    print(core_by_district.to_string())
    print(f"\n各类型数量:")
    print(df_clean["type"].value_counts().to_string())

    output_path = project_root / "outputs" / "poi" / "poi_clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {output_path}")
