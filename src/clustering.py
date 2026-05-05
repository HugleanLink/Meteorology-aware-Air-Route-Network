from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from coordinate import haversine_distance, lonlat_to_xy, build_nofly_polygons, point_in_nofly


def find_main_centers(df_clean: pd.DataFrame, config: dict) -> dict:
    cluster_cfg = config["clustering"]
    main_radius_m = cluster_cfg["main_radius_km"] * 1000
    target_coverage = cluster_cfg["target_coverage"]
    kmax = cluster_cfg["kmax"]
    near_center_m = cluster_cfg["near_center_m"]

    core_mask = df_clean["is_core"].values
    weights = df_clean["weight"].values

    x, y = lonlat_to_xy(df_clean["lon"].values, df_clean["lat"].values)
    coords_xy = np.column_stack([x, y])

    core_lons = df_clean.loc[core_mask, "lon"].values
    core_lats = df_clean.loc[core_mask, "lat"].values

    best_k = 1
    best_coverage = 0.0
    best_labels = None
    best_centroids_xy = None

    for k in range(1, kmax + 1):
        km = KMeans(n_clusters=k, random_state=config["project"]["random_state"], n_init=10)
        km.fit(coords_xy, sample_weight=weights)
        labels = km.labels_
        centroids_xy = km.cluster_centers_

        cluster_lons = np.zeros(k)
        cluster_lats = np.zeros(k)
        for c in range(k):
            mask_c = labels == c
            if mask_c.sum() > 0:
                cluster_lons[c] = np.average(df_clean.loc[mask_c, "lon"].values, weights=weights[mask_c])
                cluster_lats[c] = np.average(df_clean.loc[mask_c, "lat"].values, weights=weights[mask_c])

        covered = 0
        for lon, lat in zip(core_lons, core_lats):
            min_dist = min(
                haversine_distance(lon, lat, cluster_lons[c], cluster_lats[c])
                for c in range(k)
            )
            if min_dist <= main_radius_m:
                covered += 1

        coverage = covered / len(core_lons) if len(core_lons) > 0 else 1.0

        if coverage > best_coverage:
            best_coverage = coverage
            best_k = k
            best_labels = labels.copy()
            best_centroids_xy = centroids_xy.copy()
            # Also save the lon/lat centroids
            best_cluster_lons = cluster_lons.copy()
            best_cluster_lats = cluster_lats.copy()

        if coverage >= target_coverage:
            break

    nofly_polygons, _ = build_nofly_polygons(config)

    # Build centers DataFrame
    centers_data = []
    for c in range(best_k):
        mask_c = best_labels == c
        cluster_core_mask = mask_c & core_mask
        cent = best_centroids_xy[c]

        best_idx = None
        is_substitute = False

        if cluster_core_mask.sum() > 0:
            cluster_core_indices = np.where(cluster_core_mask)[0]
            cluster_core_coords = coords_xy[cluster_core_indices]
            dists = np.sqrt(((cluster_core_coords - cent) ** 2).sum(axis=1))
            sorted_order = dists.argsort()
            for idx_in_sorted in sorted_order:
                cand_idx = cluster_core_indices[idx_in_sorted]
                cand_row = df_clean.iloc[cand_idx]
                if not point_in_nofly(cand_row["lon"], cand_row["lat"], nofly_polygons):
                    best_idx = cand_idx
                    break
            if best_idx is None:
                # All core POIs in nofly zone, fallback to any non-nofly POI
                cluster_indices = np.where(mask_c)[0]
                cluster_coords = coords_xy[cluster_indices]
                dists = np.sqrt(((cluster_coords - cent) ** 2).sum(axis=1))
                sorted_order = dists.argsort()
                for idx_in_sorted in sorted_order:
                    cand_idx = cluster_indices[idx_in_sorted]
                    cand_row = df_clean.iloc[cand_idx]
                    if not point_in_nofly(cand_row["lon"], cand_row["lat"], nofly_polygons):
                        best_idx = cand_idx
                        is_substitute = True
                        break
                if best_idx is None:
                    # All POIs in nofly zone — fallback to closest core
                    best_idx = cluster_core_indices[dists.argsort()[0]]
                    is_substitute = True
                    print(f"警告：簇 {c} 所有点在禁飞区内，使用最近核心点作为替代")
        else:
            cluster_indices = np.where(mask_c)[0]
            cluster_coords = coords_xy[cluster_indices]
            dists = np.sqrt(((cluster_coords - cent) ** 2).sum(axis=1))
            sorted_order = dists.argsort()
            for idx_in_sorted in sorted_order:
                cand_idx = cluster_indices[idx_in_sorted]
                cand_row = df_clean.iloc[cand_idx]
                if not point_in_nofly(cand_row["lon"], cand_row["lat"], nofly_polygons):
                    best_idx = cand_idx
                    is_substitute = True
                    break
            if best_idx is None:
                best_idx = cluster_indices[dists.argsort()[0]]
                is_substitute = True

        if is_substitute and best_idx is not None:
            print(f"警告：簇 {c} 因禁飞区限制使用替代中心 (is_substitute=True)")

        row = df_clean.iloc[best_idx]
        p_i = weights[mask_c].sum()

        centers_data.append({
            "center_id": c,
            "poi_id": row["poi_id"],
            "name": row["name"],
            "lon": row["lon"],
            "lat": row["lat"],
            "cluster_id": c,
            "P_i": p_i,
            "is_substitute": is_substitute,
        })

    centers = pd.DataFrame(centers_data, columns=[
        "center_id", "poi_id", "name", "lon", "lat",
        "cluster_id", "P_i", "is_substitute",
    ])

    return {
        "centers": centers,
        "labels": best_labels,
        "k": best_k,
        "coverage": best_coverage,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_clean = pd.read_csv(
        project_root / "outputs" / "poi" / "poi_clean.csv", encoding="utf-8-sig"
    )

    # --- Diagnostics ---
    core_mask = df_clean["is_core"].values
    weights = df_clean["weight"].values
    x, y = lonlat_to_xy(df_clean["lon"].values, df_clean["lat"].values)
    coords_xy = np.column_stack([x, y])

    core_lons = df_clean.loc[core_mask, "lon"].values
    core_lats = df_clean.loc[core_mask, "lat"].values

    print("=== DIAGNOSTICS ===")
    print(f"Total POIs: {len(df_clean)}, Core POIs: {core_mask.sum()}")
    print(f"lon range: {df_clean['lon'].min():.4f} ~ {df_clean['lon'].max():.4f}")
    print(f"lat range: {df_clean['lat'].min():.4f} ~ {df_clean['lat'].max():.4f}")
    print(f"Dongli expected: lon 117.15-117.45, lat 39.05-39.25")
    in_dongli = (df_clean['lon'] >= 117.15) & (df_clean['lon'] <= 117.45) & (df_clean['lat'] >= 39.05) & (df_clean['lat'] <= 39.25)
    print(f"POIs in Dongli range: {in_dongli.sum()}/{len(df_clean)}")

    # Run KMeans with k=3 to inspect centroids
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(coords_xy, sample_weight=weights)
    for c in range(3):
        mask_c = km.labels_ == c
        clon = np.average(df_clean.loc[mask_c, "lon"].values, weights=weights[mask_c])
        clat = np.average(df_clean.loc[mask_c, "lat"].values, weights=weights[mask_c])
        print(f"  Cluster {c} centroid: lon={clon:.4f}, lat={clat:.4f}  (n={mask_c.sum()})")

    # Sample core POI distance to nearest centroid
    if len(core_lons) > 0:
        sample_lon, sample_lat = core_lons[0], core_lats[0]
        min_d = min(
            haversine_distance(sample_lon, sample_lat,
                np.average(df_clean.loc[km.labels_ == c, "lon"].values, weights=weights[km.labels_ == c]),
                np.average(df_clean.loc[km.labels_ == c, "lat"].values, weights=weights[km.labels_ == c]))
            for c in range(3)
        )
        print(f"\nSample core POI (idx 0): lon={sample_lon:.4f}, lat={sample_lat:.4f}")
        print(f"  Distance to nearest centroid: {min_d:.0f} m  (threshold: {config['clustering']['main_radius_km']*1000:.0f} m)")

    # Coverage with k=3
    cluster_lons = np.zeros(3)
    cluster_lats = np.zeros(3)
    for c in range(3):
        mask_c = km.labels_ == c
        cluster_lons[c] = np.average(df_clean.loc[mask_c, "lon"].values, weights=weights[mask_c])
        cluster_lats[c] = np.average(df_clean.loc[mask_c, "lat"].values, weights=weights[mask_c])
    radius_m = config['clustering']['main_radius_km'] * 1000
    covered = 0
    for lon, lat in zip(core_lons, core_lats):
        min_dist = min(haversine_distance(lon, lat, cluster_lons[c], cluster_lats[c]) for c in range(3))
        if min_dist <= radius_m:
            covered += 1
    print(f"\nk=3 coverage: {covered}/{len(core_lons)} = {covered/len(core_lons)*100:.1f}%")
    print("=== END DIAGNOSTICS ===\n")

    # --- Actual run ---
    result = find_main_centers(df_clean, config)

    print(f"Main centers: {result['k']}")
    print(f"Coverage: {result['coverage']:.2%}")
    print(f"\nCenter coordinates:")
    for _, row in result["centers"].iterrows():
        print(f"  center_id={row['center_id']}: lon={row['lon']:.4f}, lat={row['lat']:.4f}  "
              f"P_i={row['P_i']:.1f}  sub={row['is_substitute']}  name={row['name']}")

    output_path = project_root / "outputs" / "network" / "main_centers.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["centers"].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_path}")
