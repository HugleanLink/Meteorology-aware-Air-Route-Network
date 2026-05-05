from pathlib import Path

import folium
import pandas as pd
import yaml

from coordinate import build_nofly_polygons


def plot_network(config: dict,
                 damage_result=None,
                 recovery_edges=None,
                 show_core_poi=False,
                 show_auxiliary_poi=False,
                 highlight_district=None,
                 show_trunk=True,
                 show_branch=True,
                 show_centers=True,
                 show_subcenters=True) -> folium.Map:
    project_root = Path(__file__).resolve().parents[1]
    net_dir = project_root / "outputs" / "network"
    poi_path = project_root / "outputs" / "poi" / "poi_clean.csv"
    viz_cfg = config["visualization"]

    centers = pd.read_csv(net_dir / "main_centers.csv")
    trunk = pd.read_csv(net_dir / "trunk_edges.csv")
    branches = pd.read_csv(net_dir / "branch_edges.csv")
    poi = pd.read_csv(poi_path, encoding="utf-8-sig")

    pending_path = net_dir / "pending_points.csv"
    pending = pd.read_csv(pending_path) if pending_path.exists() else pd.DataFrame()

    subcenters_path = net_dir / "subcenters.csv"
    subcenters = pd.read_csv(subcenters_path) if subcenters_path.exists() else pd.DataFrame()

    map_center = [centers["lat"].mean(), centers["lon"].mean()]
    m = folium.Map(location=map_center, zoom_start=12, tiles=None)
    folium.TileLayer("OpenStreetMap").add_to(m)

    # --- Feature Groups ---
    fg_trunk = folium.FeatureGroup(name="主干航路")
    fg_branch = folium.FeatureGroup(name="支线")
    fg_centers = folium.FeatureGroup(name="主中心")
    fg_core = folium.FeatureGroup(name="核心需求点", show=show_core_poi)
    fg_aux = folium.FeatureGroup(name="辅助需求点", show=show_auxiliary_poi)
    fg_pending = folium.FeatureGroup(name="待处理点")
    fg_nofly = folium.FeatureGroup(name="禁飞区")

    # --- Trunk edges ---
    def _trunk_edge_id(ci, cj):
        return f"{ci}-{cj}"

    damage_map = {}
    if damage_result is not None:
        trunk_status = damage_result.get("trunk_edge_status")
        if trunk_status is not None:
            for _, row in trunk_status.iterrows():
                damage_map[_trunk_edge_id(int(row["center_i"]), int(row["center_j"]))] = row

    recovery_set = set(recovery_edges) if recovery_edges else set()

    for _, edge in trunk.iterrows():
        ci, cj = int(edge["center_i"]), int(edge["center_j"])
        row_i = centers[centers["center_id"] == ci].iloc[0]
        row_j = centers[centers["center_id"] == cj].iloc[0]
        coords = [[row_i["lat"], row_i["lon"]], [row_j["lat"], row_j["lon"]]]
        d_km = edge["d_ij"] / 1000
        is_red = edge.get("is_redundant", False)
        eid = _trunk_edge_id(ci, cj)

        dmg_info = damage_map.get(eid)
        is_recovery = eid in recovery_set

        if is_recovery:
            color = viz_cfg.get("recovery_edge_color", "purple")
            weight = 5
            dash = "10, 5"
            label = "恢复边"
        elif dmg_info is not None:
            status = dmg_info["damage_status"]
            if status == "failed":
                color = viz_cfg.get("failed_edge_color", "red")
            elif status == "affected":
                color = viz_cfg.get("affected_edge_color", "orange")
            else:
                color = viz_cfg.get("normal_edge_color", "blue")
            weight = 4
            dash = None
            label = {"failed": "失效", "affected": "受影响", "normal": "正常"}.get(status, status)
        else:
            color = viz_cfg.get("normal_edge_color", "blue")
            weight = 4
            dash = None
            label = "基干边" if not is_red else "冗余边"

        tooltip_parts = [f"{row_i['name']} — {row_j['name']}",
                         f"距离: {d_km:.2f} km",
                         f"状态: {label}"]
        if dmg_info is not None:
            tooltip_parts.append(f"fail_prob: {dmg_info['fail_prob']:.3f}")
            tooltip_parts.append(f"affected_prob: {dmg_info['affected_prob']:.3f}")
        tooltip = "<br>".join(tooltip_parts)

        kwargs = dict(color=color, weight=weight, opacity=0.9, tooltip=tooltip)
        if dash:
            kwargs["dash_array"] = dash
        folium.PolyLine(coords, **kwargs).add_to(fg_trunk)

    # --- Branch edges ---
    branch_damage_map = {}
    if damage_result is not None:
        branch_status = damage_result.get("branch_edge_status")
        if branch_status is not None:
            for _, row in branch_status.iterrows():
                branch_damage_map[str(row["edge_id"])] = row

    for _, br in branches.iterrows():
        connect_to = br["connect_to"]
        ctype = br["connect_type"]
        if ctype == "center":
            cid = int(connect_to)
            c_row = centers[centers["center_id"] == cid].iloc[0]
            clon, clat = c_row["lon"], c_row["lat"]
        else:
            parts = str(connect_to).split("-")
            ci, cj = int(parts[0]), int(parts[1])
            r_i = centers[centers["center_id"] == ci].iloc[0]
            r_j = centers[centers["center_id"] == cj].iloc[0]
            clon = (r_i["lon"] + r_j["lon"]) / 2
            clat = (r_i["lat"] + r_j["lat"]) / 2
        coords = [[br["lat"], br["lon"]], [clat, clon]]

        br_eid = f"B_{br['poi_id']}"
        bdmg = branch_damage_map.get(br_eid)
        is_br_recovery = br_eid in recovery_set

        if is_br_recovery:
            color = viz_cfg.get("recovery_edge_color", "purple")
            weight = 3
            opacity = 0.9
            dash = "8, 4"
            folium.PolyLine(coords, color=color, weight=weight,
                            opacity=opacity, dash_array=dash).add_to(fg_branch)
        elif bdmg is not None and bdmg["damage_status"] == "failed":
            color = viz_cfg.get("failed_edge_color", "red")
            folium.PolyLine(coords, color=color, weight=2, opacity=0.7).add_to(fg_branch)
        else:
            folium.PolyLine(coords, color="gray", weight=1.5, opacity=0.6).add_to(fg_branch)

    # --- Main centers ---
    for _, c in centers.iterrows():
        popup = f"<b>{c['name']}</b><br>P_i: {c['P_i']:.1f}<br>经度: {c['lon']:.4f}<br>纬度: {c['lat']:.4f}"
        folium.CircleMarker(
            location=[c["lat"], c["lon"]], radius=10,
            color="red", fill=True, fill_color="red", fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(fg_centers)

    # --- Subcenters ---
    if show_subcenters and not subcenters.empty:
        fg_sub = folium.FeatureGroup(name="次中心")
        for _, sc in subcenters.iterrows():
            folium.CircleMarker(
                location=[sc["lat"], sc["lon"]], radius=7,
                color="orange", fill=True, fill_color="orange", fill_opacity=0.7,
            ).add_to(fg_sub)
        fg_sub.add_to(m)

    # --- Core POIs (non-center) ---
    center_poi_ids = set(centers["poi_id"].values)
    core = poi[poi["is_core"] & ~poi["poi_id"].isin(center_poi_ids)]
    if highlight_district:
        core = core[core["district"] == highlight_district]
    for _, c in core.iterrows():
        folium.CircleMarker(
            location=[c["lat"], c["lon"]], radius=4,
            color="blue", fill=True, fill_color="blue", fill_opacity=0.5,
        ).add_to(fg_core)

    # --- Auxiliary POIs (non-core) ---
    aux = poi[~poi["is_core"]]
    if highlight_district:
        aux = aux[aux["district"] == highlight_district]
    for _, a in aux.iterrows():
        folium.CircleMarker(
            location=[a["lat"], a["lon"]], radius=3,
            color="green", fill=True, fill_color="green", fill_opacity=0.4,
        ).add_to(fg_aux)

    # --- Pending points ---
    if not pending.empty:
        for _, p in pending.iterrows():
            popup = f"<b>{p['name']}</b><br>d_min: {p['d_min']:.0f} m"
            folium.Marker(
                location=[p["lat"], p["lon"]],
                icon=folium.DivIcon(
                    html='<div style="color:red;font-size:14px;font-weight:bold">✕</div>',
                    icon_size=(14, 14), icon_anchor=(7, 7),
                ),
                popup=folium.Popup(popup, max_width=200),
            ).add_to(fg_pending)

    # --- No-fly zones ---
    _, nofly_meta = build_nofly_polygons(config)
    for meta_item in nofly_meta:
        if meta_item.get("is_real"):
            coords = [(lat, lon) for lon, lat in meta_item["coords"]]
            popup_text = f"{meta_item['name']}<br>（OSM真实边界）"
            folium.Polygon(
                locations=coords,
                color="red",
                fill=True,
                fill_opacity=0.15,
                weight=2,
                popup=folium.Popup(popup_text, max_width=250),
            ).add_to(fg_nofly)
        else:
            popup_text = f"{meta_item['name']}<br>半径: {meta_item['radius_km']} km<br>（近似圆形）"
            folium.Circle(
                location=[meta_item["center_lat"], meta_item["center_lon"]],
                radius=meta_item["radius_km"] * 1000,
                color="red",
                fill=True,
                fill_opacity=0.15,
                weight=2,
                popup=folium.Popup(popup_text, max_width=250),
            ).add_to(fg_nofly)

    # --- Add all feature groups to map ---
    if show_trunk:
        fg_trunk.add_to(m)
    if show_branch:
        fg_branch.add_to(m)
    if show_centers:
        fg_centers.add_to(m)
    if show_core_poi:
        fg_core.add_to(m)
    if show_auxiliary_poi:
        fg_aux.add_to(m)
    fg_pending.add_to(m)
    fg_nofly.add_to(m)

    folium.LayerControl().add_to(m)

    return m


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    m = plot_network(config)

    out_dir = project_root / "outputs" / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    m.save(str(out_dir / "network_map.html"))
    print("地图已保存")
