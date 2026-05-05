import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_era5_path(data_dir="data/raw/era5"):
    x_path = Path("X:/data/raw/era5")
    if x_path.exists():
        return x_path
    rel_path = _PROJECT_ROOT / data_dir
    if rel_path.exists():
        return rel_path
    abs_path = Path(data_dir)
    if abs_path.exists():
        return abs_path
    raise FileNotFoundError(f"找不到 ERA5 数据目录: {data_dir}")


def load_era5(data_dir="data/raw/era5") -> pd.DataFrame:
    """
    Read all .nc files from data_dir, merge into a single DataFrame.

    Extracts u10 (eastward wind), v10 (northward wind), tp (total precipitation).
    Computes wind_speed = sqrt(u10^2 + v10^2).
    Converts tp from m/hour to mm/day: tp_mm_day = tp * 1000 * 24.

    Returns DataFrame with columns: time, lat, lon, u10, v10, wind_speed, tp_mm_day
    """
    era5_path = _resolve_era5_path(data_dir)
    nc_files = sorted(era5_path.glob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {era5_path}")

    print(f"找到 {len(nc_files)} 个 NetCDF 文件:")
    for f in nc_files:
        print(f"  - {f.name}")

    dfs = []
    for f in nc_files:
        ds = xr.open_dataset(f)
        ds = ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
        ds = ds[["u10", "v10", "tp"]]
        ds = ds.drop_vars(["number", "expver"], errors="ignore")
        df = ds.to_dataframe().reset_index()
        dfs.append(df)
        ds.close()

    df_era5 = pd.concat(dfs, ignore_index=True)

    df_era5["wind_speed"] = np.sqrt(df_era5["u10"] ** 2 + df_era5["v10"] ** 2)
    df_era5["tp_mm_day"] = df_era5["tp"] * 1000 * 24

    df_era5 = df_era5[["time", "lat", "lon", "u10", "v10", "wind_speed", "tp_mm_day"]]
    df_era5 = df_era5.sort_values("time").reset_index(drop=True)

    print(f"ERA5 数据加载完成: {len(df_era5)} 条记录")
    print(f"时间范围: {df_era5['time'].min()} ~ {df_era5['time'].max()}")
    lat_vals = df_era5["lat"].unique()
    lon_vals = df_era5["lon"].unique()
    print(f"空间范围: lat [{lat_vals.min()}, {lat_vals.max()}], lon [{lon_vals.min()}, {lon_vals.max()}]")
    print(f"网格: {len(lat_vals)} × {len(lon_vals)}, 时间步: {len(df_era5['time'].unique())}")

    return df_era5


def classify_weather_pool(df_era5: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Classify each ERA5 record into a weather sample pool.

    Classification (priority order):
      风雨复合: wind_speed >= wind_p90 AND tp_mm_day >= rain_p90
      极端风:   wind_speed >= wind_p90
      极端雨:   tp_mm_day >= rain_p90
      常规:     otherwise

    Returns DataFrame with added 'weather_pool' column.
    """
    extreme_pct = config["weather"]["extreme_percentile"]

    wind_p90 = df_era5["wind_speed"].quantile(extreme_pct)
    rain_p90 = df_era5["tp_mm_day"].quantile(extreme_pct)

    print(f"\n气象样本池分类 (P{int(extreme_pct*100)} 阈值):")
    print(f"  风速 P{int(extreme_pct*100)}: {wind_p90:.4f} m/s")
    print(f"  降水 P{int(extreme_pct*100)}: {rain_p90:.4f} mm/day")

    def _classify(row):
        high_wind = row["wind_speed"] >= wind_p90
        high_rain = row["tp_mm_day"] >= rain_p90
        if high_wind and high_rain:
            return "风雨复合"
        elif high_wind:
            return "极端风"
        elif high_rain:
            return "极端雨"
        else:
            return "常规"

    df = df_era5.copy()
    df["weather_pool"] = df.apply(_classify, axis=1)

    total = len(df)
    for pool in ["风雨复合", "极端风", "极端雨", "常规"]:
        count = (df["weather_pool"] == pool).sum()
        print(f"  {pool}: {count:6d} ({count/total*100:.2f}%)")

    return df


def _build_era5_grids(df_era5):
    """Convert ERA5 DataFrame to 3D numpy arrays indexed by (time, lat, lon)."""
    lats = np.sort(df_era5["lat"].unique())
    lons = np.sort(df_era5["lon"].unique())
    times = np.sort(df_era5["time"].unique())

    n_times = len(times)
    n_lats = len(lats)
    n_lons = len(lons)

    df_sorted = df_era5.sort_values(["time", "lat", "lon"]).reset_index(drop=True)

    wind_grid = df_sorted["wind_speed"].values.reshape(n_times, n_lats, n_lons)
    rain_grid = df_sorted["tp_mm_day"].values.reshape(n_times, n_lats, n_lons)

    pool_codes_map = {"风雨复合": 30, "极端雨": 20, "极端风": 10, "常规": 0}
    pool_codes = np.array([pool_codes_map.get(p, 0) for p in df_sorted["weather_pool"]])
    pool_grid = pool_codes.reshape(n_times, n_lats, n_lons)

    return times, lats, lons, wind_grid, rain_grid, pool_grid


def _find_nearest_grid_indices(interp_lons, interp_lats, grid_lons, grid_lats):
    """Find nearest grid indices for an array of interpolation points."""
    lon_idx = np.array([np.argmin(np.abs(grid_lons - lon)) for lon in interp_lons])
    lat_idx = np.array([np.argmin(np.abs(grid_lats - lat)) for lat in interp_lats])
    return lat_idx, lon_idx


def build_edge_weather_samples(trunk_edges, config, df_era5_classified):
    """
    Build weather samples for all trunk edges by interpolating spatial points
    and aggregating ERA5 data per time step using vectorized numpy operations.

    Saves result to data/processed/edge_weather_samples.parquet.
    Returns the resulting DataFrame.
    """
    interval_m = config["weather"]["edge_sample_interval_m"]
    min_points = config["weather"]["edge_min_sample_points"]

    centers_path = _PROJECT_ROOT / "outputs" / "network" / "main_centers.csv"
    centers = pd.read_csv(centers_path)
    center_coords = centers.set_index("center_id")[["lon", "lat"]].to_dict("index")

    edges = trunk_edges.copy()
    edges["center_i_lon"] = edges["center_i"].map(lambda i: center_coords[i]["lon"])
    edges["center_i_lat"] = edges["center_i"].map(lambda i: center_coords[i]["lat"])
    edges["center_j_lon"] = edges["center_j"].map(lambda i: center_coords[i]["lon"])
    edges["center_j_lat"] = edges["center_j"].map(lambda i: center_coords[i]["lat"])
    edges["edge_id"] = edges.apply(lambda r: f"{int(r['center_i'])}-{int(r['center_j'])}", axis=1)

    n_edges = len(edges)
    n_times = len(df_era5_classified["time"].unique())
    print(f"\n构建边气象样本: {n_edges} 条主干边 × {n_times} 时间步 = {n_edges * n_times} 预计样本")

    print("  构建 ERA5 3D 网格 ...")
    times, grid_lats, grid_lons, wind_grid, rain_grid, pool_grid = _build_era5_grids(df_era5_classified)

    def _pool_code_to_label(code):
        if code >= 30:
            return "风雨复合"
        elif code >= 20:
            return "极端雨"
        elif code >= 10:
            return "极端风"
        return "常规"

    all_frames = []
    for idx, row in edges.iterrows():
        lon1, lat1 = row["center_i_lon"], row["center_i_lat"]
        lon2, lat2 = row["center_j_lon"], row["center_j_lat"]
        d_ij = row["d_ij"]
        edge_id = row["edge_id"]

        n_points = max(min_points, int(math.ceil(d_ij / interval_m)))
        lons_interp = np.linspace(lon1, lon2, n_points)
        lats_interp = np.linspace(lat1, lat2, n_points)

        lat_idx, lon_idx = _find_nearest_grid_indices(lons_interp, lats_interp, grid_lons, grid_lats)

        edge_wind = wind_grid[:, lat_idx, lon_idx].mean(axis=1)
        edge_rain = rain_grid[:, lat_idx, lon_idx].mean(axis=1)
        edge_pool_code = pool_grid[:, lat_idx, lon_idx].max(axis=1)

        df_edge = pd.DataFrame({
            "time": times,
            "edge_id": edge_id,
            "wind_speed": edge_wind,
            "tp_mm_day": edge_rain,
            "weather_pool": [_pool_code_to_label(c) for c in edge_pool_code],
        })

        all_frames.append(df_edge)
        print(f"  处理边 {idx+1}/{n_edges} (edge_id={edge_id}, {n_points} space points)")

    result = pd.concat(all_frames, ignore_index=True)

    output_dir = _PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "edge_weather_samples.parquet"
    result.to_parquet(output_path, index=False)

    print(f"\n边气象样本构建完成: {len(result)} 条记录")
    print(f"保存至: {output_path}")

    return result


def _project_point_to_segment_lonlat(px, py, ax, ay, bx, by):
    """Project point P onto segment AB, all in lon/lat (OK for short distances)."""
    abx, aby = bx - ax, by - ay
    dot_ab_ab = abx * abx + aby * aby
    if dot_ab_ab < 1e-12:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / dot_ab_ab
    t = max(0.0, min(1.0, t))
    return ax + t * abx, ay + t * aby


def _sample_edge_weather(row, times, grid_lats, grid_lons,
                         wind_grid, rain_grid, pool_grid,
                         lon1, lat1, lon2, lat2, d_m, interval_m, min_points):
    """Sample weather for one edge (trunk or branch) across all time steps."""
    n_points = max(min_points, int(math.ceil(d_m / interval_m)))
    lons_interp = np.linspace(lon1, lon2, n_points)
    lats_interp = np.linspace(lat1, lat2, n_points)
    lat_idx, lon_idx = _find_nearest_grid_indices(lons_interp, lats_interp, grid_lons, grid_lats)
    edge_wind = wind_grid[:, lat_idx, lon_idx].mean(axis=1)
    edge_rain = rain_grid[:, lat_idx, lon_idx].mean(axis=1)
    edge_pool_code = pool_grid[:, lat_idx, lon_idx].max(axis=1)
    return edge_wind, edge_rain, edge_pool_code


def build_branch_weather_samples(branch_edges, config, df_era5_classified):
    """
    Build weather samples for all branch edges.
    Returns DataFrame with columns: time, edge_id, wind_speed, tp_mm_day,
    weather_pool, edge_type
    """
    interval_m = config["weather"]["edge_sample_interval_m"]
    min_points = max(config["weather"].get("branch_min_sample_points", 3), 3)
    centers = pd.read_csv(_PROJECT_ROOT / "outputs" / "network" / "main_centers.csv")
    center_coords = centers.set_index("center_id")[["lon", "lat"]].to_dict("index")

    trunk_edges = pd.read_csv(_PROJECT_ROOT / "outputs" / "network" / "trunk_edges.csv")
    trunk_edge_map = {}
    for _, te in trunk_edges.iterrows():
        tid = f"{int(te['center_i'])}-{int(te['center_j'])}"
        trunk_edge_map[tid] = (int(te["center_i"]), int(te["center_j"]))

    branches = branch_edges.copy()
    n_branches = len(branches)
    n_times = len(df_era5_classified["time"].unique())
    print(f"\n构建支线气象样本: {n_branches} 条支线 × {n_times} 时间步")

    print("  构建 ERA5 3D 网格 ...")
    times, grid_lats, grid_lons, wind_grid, rain_grid, pool_grid = _build_era5_grids(df_era5_classified)

    def _pool_code_to_label(code):
        if code >= 30:
            return "风雨复合"
        elif code >= 20:
            return "极端雨"
        elif code >= 10:
            return "极端风"
        return "常规"

    all_frames = []
    for idx, row in branches.iterrows():
        poi_id = row["poi_id"]
        lon1, lat1 = float(row["lon"]), float(row["lat"])

        if row["connect_type"] == "center":
            cid = int(row["connect_to"])
            lon2, lat2 = center_coords[cid]["lon"], center_coords[cid]["lat"]
        else:
            # edge type: connect_to is edge_id like "3-8"
            eid = row["connect_to"]
            ci, cj = trunk_edge_map[eid]
            ci_lon, ci_lat = center_coords[ci]["lon"], center_coords[ci]["lat"]
            cj_lon, cj_lat = center_coords[cj]["lon"], center_coords[cj]["lat"]
            lon2, lat2 = _project_point_to_segment_lonlat(
                lon1, lat1, ci_lon, ci_lat, cj_lon, cj_lat
            )

        d_m = float(row["d_branch"])
        edge_wind, edge_rain, edge_pool_code = _sample_edge_weather(
            row, times, grid_lats, grid_lons, wind_grid, rain_grid, pool_grid,
            lon1, lat1, lon2, lat2, d_m, interval_m, min_points
        )

        edge_id = f"B_{poi_id}"
        df_edge = pd.DataFrame({
            "time": times,
            "edge_id": edge_id,
            "wind_speed": edge_wind.astype(np.float32),
            "tp_mm_day": edge_rain.astype(np.float32),
            "weather_pool": [_pool_code_to_label(c) for c in edge_pool_code],
            "edge_type": 1,
        })
        all_frames.append(df_edge)
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  处理支线 {idx+1}/{n_branches} (edge_id={edge_id})")

    result = pd.concat(all_frames, ignore_index=True)
    print(f"支线气象样本构建完成: {len(result)} 条记录")
    return result


def build_all_edge_weather_samples(trunk_edges, branch_edges, config, df_era5_classified):
    """Build and merge trunk + branch weather samples."""
    print("\n" + "=" * 60)
    print("构建主干边气象样本")
    print("=" * 60)
    trunk_samples = build_edge_weather_samples(trunk_edges, config, df_era5_classified)
    trunk_samples["edge_type"] = 0

    print("\n" + "=" * 60)
    print("构建支线气象样本")
    print("=" * 60)
    branch_samples = build_branch_weather_samples(branch_edges, config, df_era5_classified)

    all_samples = pd.concat([trunk_samples, branch_samples], ignore_index=True)

    output_dir = _PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "edge_weather_samples_all.parquet"
    all_samples.to_parquet(output_path, index=False)

    print(f"\n合并完成:")
    print(f"  主干边样本: {len(trunk_samples):>10,}")
    print(f"  支线样本:   {len(branch_samples):>10,}")
    print(f"  总样本:     {len(all_samples):>10,}")
    print(f"  保存至: {output_path}")
    return all_samples


if __name__ == "__main__":
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("任务1: load_era5")
    print("=" * 60)
    df_era5 = load_era5()
    print(f"\n总记录数: {len(df_era5)}")
    print(f"时间范围: {df_era5['time'].min()} ~ {df_era5['time'].max()}")

    print("\n" + "=" * 60)
    print("任务2: classify_weather_pool")
    print("=" * 60)
    df_classified = classify_weather_pool(df_era5, cfg)

    print("\n" + "=" * 60)
    print("任务3: build_edge_weather_samples")
    print("=" * 60)
    trunk_path = _PROJECT_ROOT / "outputs" / "network" / "trunk_edges.csv"
    trunk_edges = pd.read_csv(trunk_path)
    print(f"读取主干边: {len(trunk_edges)} 条")
    print(f"  其中基干: {(~trunk_edges['is_redundant']).sum()} 条, 冗余: {trunk_edges['is_redundant'].sum()} 条")

    edge_samples = build_edge_weather_samples(trunk_edges, cfg, df_classified)

    print("\n" + "=" * 60)
    print("最终结果预览")
    print("=" * 60)
    print(f"总样本数: {len(edge_samples)}")
    print(f"唯一边数: {edge_samples['edge_id'].nunique()}")
    print(f"时间步数: {edge_samples['time'].nunique()}")
    print(f"各池分布:")
    for pool in ["风雨复合", "极端风", "极端雨", "常规"]:
        count = (edge_samples["weather_pool"] == pool).sum()
        print(f"  {pool}: {count} ({count/len(edge_samples)*100:.2f}%)")
    print(edge_samples.head(10))
