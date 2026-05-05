from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from coordinate import haversine_distance, build_nofly_polygons, edge_intersects_nofly


def _point_to_segment_distance(poi_lon, poi_lat, seg_lon1, seg_lat1, seg_lon2, seg_lat2,
                                interval_m=500, min_samples=5) -> float:
    seg_length = haversine_distance(seg_lon1, seg_lat1, seg_lon2, seg_lat2)
    n_samples = max(min_samples, int(seg_length / interval_m) + 1)

    best_dist = float("inf")
    for t in np.linspace(0, 1, n_samples):
        slon = seg_lon1 + t * (seg_lon2 - seg_lon1)
        slat = seg_lat1 + t * (seg_lat2 - seg_lat1)
        d = haversine_distance(poi_lon, poi_lat, slon, slat)
        if d < best_dist:
            best_dist = d
    return best_dist


def connect_branches(df_clean: pd.DataFrame, centers: pd.DataFrame,
                     trunk_edges: pd.DataFrame, config: dict) -> dict:
    branch_cfg = config["branch"]
    r_branch = branch_cfg["max_branch_distance_km"] * 1000
    near_center_m = config["clustering"]["near_center_m"]
    interval_m = config["weather"]["edge_sample_interval_m"]
    min_samples = config["weather"]["edge_min_sample_points"]

    core = df_clean[df_clean["is_core"]].copy()

    center_lons = centers["lon"].values
    center_lats = centers["lat"].values
    center_ids = centers["center_id"].values

    already_connected = []
    to_connect = []

    for _, poi in core.iterrows():
        d_to_centers = [haversine_distance(poi["lon"], poi["lat"], clon, clat)
                        for clon, clat in zip(center_lons, center_lats)]
        min_d_center = min(d_to_centers)
        nearest_center = center_ids[int(np.argmin(d_to_centers))]

        if min_d_center <= near_center_m:
            already_connected.append({
                "poi_id": poi["poi_id"], "name": poi["name"],
                "lon": poi["lon"], "lat": poi["lat"],
                "nearest_center": nearest_center, "d_center": min_d_center,
            })
        else:
            to_connect.append(poi)

    already_df = pd.DataFrame(already_connected,
        columns=["poi_id", "name", "lon", "lat", "nearest_center", "d_center"])

    nofly_polygons, _ = build_nofly_polygons(config)
    branches = []
    pending = []
    nofly_filtered = 0

    for poi in to_connect:
        d_to_centers = [haversine_distance(poi["lon"], poi["lat"], clon, clat)
                        for clon, clat in zip(center_lons, center_lats)]
        min_d_center = min(d_to_centers)
        nearest_center = center_ids[int(np.argmin(d_to_centers))]

        min_d_edge = float("inf")
        nearest_edge = None
        nearest_edge_coords = None
        for _, edge in trunk_edges.iterrows():
            ci, cj = int(edge["center_i"]), int(edge["center_j"])
            c_row_i = centers[centers["center_id"] == ci].iloc[0]
            c_row_j = centers[centers["center_id"] == cj].iloc[0]
            d = _point_to_segment_distance(
                poi["lon"], poi["lat"],
                c_row_i["lon"], c_row_i["lat"],
                c_row_j["lon"], c_row_j["lat"],
                interval_m, min_samples,
            )
            if d < min_d_edge:
                min_d_edge = d
                nearest_edge = (ci, cj)
                nearest_edge_coords = (c_row_i["lon"], c_row_i["lat"],
                                       c_row_j["lon"], c_row_j["lat"])

        if min(min_d_center, min_d_edge) <= r_branch:
            # Check no-fly zone before creating branch
            crosses_nofly = False
            if min_d_center <= min_d_edge:
                c_row = centers[centers["center_id"] == nearest_center].iloc[0]
                crosses_nofly = edge_intersects_nofly(
                    poi["lon"], poi["lat"], c_row["lon"], c_row["lat"], nofly_polygons)
            else:
                crosses_nofly = edge_intersects_nofly(
                    poi["lon"], poi["lat"],
                    nearest_edge_coords[0], nearest_edge_coords[1],
                    nofly_polygons)

            if crosses_nofly:
                pending.append({
                    "poi_id": poi["poi_id"], "name": poi["name"],
                    "lon": poi["lon"], "lat": poi["lat"],
                    "d_min": min(min_d_center, min_d_edge),
                    "reason": "nofly",
                })
                nofly_filtered += 1
            elif min_d_center <= min_d_edge:
                branches.append({
                    "poi_id": poi["poi_id"], "name": poi["name"],
                    "lon": poi["lon"], "lat": poi["lat"],
                    "connect_to": int(nearest_center),
                    "connect_type": "center",
                    "d_branch": min_d_center,
                })
            else:
                branches.append({
                    "poi_id": poi["poi_id"], "name": poi["name"],
                    "lon": poi["lon"], "lat": poi["lat"],
                    "connect_to": f"{nearest_edge[0]}-{nearest_edge[1]}",
                    "connect_type": "edge",
                    "d_branch": min_d_edge,
                })
        else:
            pending.append({
                "poi_id": poi["poi_id"], "name": poi["name"],
                "lon": poi["lon"], "lat": poi["lat"],
                "d_min": min(min_d_center, min_d_edge),
                "reason": "distance",
            })

    if nofly_filtered > 0:
        print(f"因禁飞区被过滤的支线数量: {nofly_filtered}")

    branches_df = pd.DataFrame(branches,
        columns=["poi_id", "name", "lon", "lat", "connect_to", "connect_type", "d_branch"])
    pending_df = pd.DataFrame(pending,
        columns=["poi_id", "name", "lon", "lat", "d_min", "reason"])

    return {
        "branches": branches_df,
        "pending": pending_df,
        "already_connected": already_df,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_clean = pd.read_csv(project_root / "outputs" / "poi" / "poi_clean.csv", encoding="utf-8-sig")
    centers = pd.read_csv(project_root / "outputs" / "network" / "main_centers.csv")
    trunk_edges = pd.read_csv(project_root / "outputs" / "network" / "trunk_edges.csv")

    result = connect_branches(df_clean, centers, trunk_edges, config)

    print(f"Already connected: {len(result['already_connected'])}")
    print(f"New branches: {len(result['branches'])}")
    print(f"Pending: {len(result['pending'])}")

    out_dir = project_root / "outputs" / "network"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["branches"].to_csv(out_dir / "branch_edges.csv", index=False, encoding="utf-8-sig")
    result["pending"].to_csv(out_dir / "pending_points.csv", index=False, encoding="utf-8-sig")
    print(f"Saved branches and pending to {out_dir}")
