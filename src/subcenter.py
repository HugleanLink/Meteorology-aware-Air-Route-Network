from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from coordinate import haversine_distance, lonlat_to_xy


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


def generate_subcenters(pending: pd.DataFrame, df_clean: pd.DataFrame,
                        centers: pd.DataFrame, trunk_edges: pd.DataFrame,
                        config: dict) -> dict:
    cluster_cfg = config["clustering"]
    sub_radius_m = cluster_cfg["sub_radius_km"] * 1000
    sub_target = cluster_cfg["sub_target_coverage"]
    trigger_count = cluster_cfg["sub_trigger_pending_count"]

    center_lons = centers["lon"].values
    center_lats = centers["lat"].values
    center_ids = centers["center_id"].values

    # Group pending by nearest center
    groups = {}
    for _, poi in pending.iterrows():
        dists = [haversine_distance(poi["lon"], poi["lat"], clon, clat)
                 for clon, clat in zip(center_lons, center_lats)]
        nc = center_ids[int(np.argmin(dists))]
        groups.setdefault(nc, []).append(poi)

    subcenters_all = []
    sub_edges_all = []
    sub_branches_all = []
    triggered_centers = 0
    subcenter_id_counter = 0

    for parent_cid, group_pois in groups.items():
        if len(group_pois) < trigger_count:
            continue

        triggered_centers += 1
        group_df = pd.DataFrame(group_pois)
        group_lons = group_df["lon"].values
        group_lats = group_df["lat"].values
        group_weights = np.ones(len(group_df))

        x, y = lonlat_to_xy(group_lons, group_lats)
        coords_xy = np.column_stack([x, y])

        best_k = 1
        best_labels = None
        best_centroids_xy = None
        best_cluster_lons = None
        best_cluster_lats = None

        for k in range(1, len(group_df) + 1):
            km = KMeans(n_clusters=k, random_state=config["project"]["random_state"], n_init=10)
            km.fit(coords_xy, sample_weight=group_weights)
            labels = km.labels_
            centroids_xy = km.cluster_centers_

            c_lons = np.zeros(k)
            c_lats = np.zeros(k)
            for c in range(k):
                mask_c = labels == c
                if mask_c.sum() > 0:
                    c_lons[c] = group_lons[mask_c].mean()
                    c_lats[c] = group_lats[mask_c].mean()

            covered = 0
            for lon, lat in zip(group_lons, group_lats):
                min_d = min(haversine_distance(lon, lat, c_lons[c], c_lats[c]) for c in range(k))
                if min_d <= sub_radius_m:
                    covered += 1
            coverage = covered / len(group_lons)

            if coverage > (best_k_cov := 0.0 if best_labels is None else best_k_cov):
                # track best coverage
                pass

            if coverage >= sub_target:
                best_k = k
                best_labels = labels.copy()
                best_centroids_xy = centroids_xy.copy()
                best_cluster_lons = c_lons.copy()
                best_cluster_lats = c_lats.copy()
                break
            else:
                # Track best so far
                if best_labels is None or coverage > getattr(best_k_tracker := None, '__coverage', 0):
                    pass
                best_k = k
                best_labels = labels.copy()
                best_centroids_xy = centroids_xy.copy()
                best_cluster_lons = c_lons.copy()
                best_cluster_lats = c_lats.copy()

        # Build subcenters for this group
        for c in range(best_k):
            mask_c = best_labels == c
            cluster_indices = np.where(mask_c)[0]
            cent = best_centroids_xy[c]
            cluster_coords = coords_xy[cluster_indices]
            dists = np.sqrt(((cluster_coords - cent) ** 2).sum(axis=1))
            best_idx_in_group = cluster_indices[dists.argmin()]
            best_poi = group_df.iloc[best_idx_in_group]

            sc_id = subcenter_id_counter
            subcenter_id_counter += 1

            subcenters_all.append({
                "subcenter_id": sc_id,
                "poi_id": best_poi["poi_id"],
                "name": best_poi["name"],
                "lon": best_poi["lon"],
                "lat": best_poi["lat"],
                "parent_center": int(parent_cid),
            })

            # Connect subcenter to network
            d_to_centers = [haversine_distance(best_poi["lon"], best_poi["lat"], clon, clat)
                            for clon, clat in zip(center_lons, center_lats)]
            min_d_center = min(d_to_centers)
            nearest_center = center_ids[int(np.argmin(d_to_centers))]

            min_d_edge = float("inf")
            nearest_edge_key = None
            for _, edge in trunk_edges.iterrows():
                ci, cj = int(edge["center_i"]), int(edge["center_j"])
                r_i = centers[centers["center_id"] == ci].iloc[0]
                r_j = centers[centers["center_id"] == cj].iloc[0]
                d = _point_to_segment_distance(
                    best_poi["lon"], best_poi["lat"],
                    r_i["lon"], r_i["lat"], r_j["lon"], r_j["lat"],
                )
                if d < min_d_edge:
                    min_d_edge = d
                    nearest_edge_key = f"{ci}-{cj}"

            if min_d_center <= min_d_edge:
                sub_edges_all.append({
                    "subcenter_id": sc_id,
                    "connect_to": int(nearest_center),
                    "connect_type": "center",
                    "d_connect": min_d_center,
                })
            else:
                sub_edges_all.append({
                    "subcenter_id": sc_id,
                    "connect_to": nearest_edge_key,
                    "connect_type": "edge",
                    "d_connect": min_d_edge,
                })

            # Branches from pending points in this cluster to subcenter
            for idx_in_group in cluster_indices:
                p_poi = group_df.iloc[idx_in_group]
                # Skip the point that became the subcenter itself
                d_sc = haversine_distance(p_poi["lon"], p_poi["lat"], best_poi["lon"], best_poi["lat"])
                sub_branches_all.append({
                    "poi_id": p_poi["poi_id"],
                    "name": p_poi["name"],
                    "lon": p_poi["lon"],
                    "lat": p_poi["lat"],
                    "subcenter_id": sc_id,
                    "d_branch": d_sc,
                })

    sub_df = pd.DataFrame(subcenters_all,
        columns=["subcenter_id", "poi_id", "name", "lon", "lat", "parent_center"])
    sub_edges_df = pd.DataFrame(sub_edges_all,
        columns=["subcenter_id", "connect_to", "connect_type", "d_connect"])
    sub_branches_df = pd.DataFrame(sub_branches_all,
        columns=["poi_id", "name", "lon", "lat", "subcenter_id", "d_branch"])

    return {
        "subcenters": sub_df,
        "sub_edges": sub_edges_df,
        "sub_branches": sub_branches_df,
        "triggered_centers": triggered_centers,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pending = pd.read_csv(project_root / "outputs" / "network" / "pending_points.csv", encoding="utf-8-sig")
    df_clean = pd.read_csv(project_root / "outputs" / "poi" / "poi_clean.csv", encoding="utf-8-sig")
    centers = pd.read_csv(project_root / "outputs" / "network" / "main_centers.csv")
    trunk_edges = pd.read_csv(project_root / "outputs" / "network" / "trunk_edges.csv")

    result = generate_subcenters(pending, df_clean, centers, trunk_edges, config)

    print(f"Triggered centers: {result['triggered_centers']}")
    print(f"Subcenters: {len(result['subcenters'])}")
    print(f"Sub-edges: {len(result['sub_edges'])}")
    # Count unique pending points connected through subcenters
    unique_connected = result["sub_branches"]["poi_id"].nunique() if not result["sub_branches"].empty else 0
    print(f"Pending points connected via subcenters: {unique_connected}")

    out_dir = project_root / "outputs" / "network"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not result["subcenters"].empty:
        result["subcenters"].to_csv(out_dir / "subcenters.csv", index=False, encoding="utf-8-sig")
    if not result["sub_edges"].empty:
        result["sub_edges"].to_csv(out_dir / "sub_edges.csv", index=False, encoding="utf-8-sig")
    if not result["sub_branches"].empty:
        result["sub_branches"].to_csv(out_dir / "sub_branches.csv", index=False, encoding="utf-8-sig")
    print(f"Saved to {out_dir}")
