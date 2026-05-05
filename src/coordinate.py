import json
import math
import urllib.parse
import urllib.request

import numpy as np
from pyproj import Proj
from shapely.geometry import Point, LineString, Polygon


def haversine_distance(lon1, lat1, lon2, lat2) -> float:
    radius = 6371000
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def lonlat_to_xy(lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_center = float(np.mean(lons))
    lat_center = float(np.mean(lats))
    projection = Proj(proj="aeqd", lat_0=lat_center, lon_0=lon_center, datum="WGS84", units="m")
    x, y = projection(lons, lats)
    return np.asarray(x), np.asarray(y)


_osm_cache: dict[int, list | None] = {}


def fetch_airport_polygon(osm_id: int) -> list | None:
    """Fetch airport boundary polygon from OpenStreetMap via Overpass API.

    Returns list of (lon, lat) tuples forming a closed polygon, or None on failure.
    Results are cached in-memory to avoid repeated API calls.
    """
    if osm_id in _osm_cache:
        return _osm_cache[osm_id]

    query = f"[out:json];\nway({osm_id});\n(._;>;);\nout body;"
    url = "https://overpass-api.de/api/interpreter?" + urllib.parse.urlencode({"data": query})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LowAltAirship/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARNING] Overpass API request failed for osm_id={osm_id}: {e}")
        _osm_cache[osm_id] = None
        return None

    elements = data.get("elements", [])
    if not elements:
        print(f"[WARNING] Overpass API returned empty response for osm_id={osm_id}")
        return None

    node_coords = {}
    way_nodes = None
    for el in elements:
        if el.get("type") == "node":
            node_coords[el["id"]] = (el["lon"], el["lat"])
        elif el.get("type") == "way" and el.get("id") == osm_id:
            way_nodes = el.get("nodes", [])

    if not way_nodes:
        print(f"[WARNING] No way with id={osm_id} found in Overpass response")
        return None

    coords = []
    for nid in way_nodes:
        if nid in node_coords:
            coords.append(node_coords[nid])
        else:
            print(f"[WARNING] Node {nid} referenced by way {osm_id} not found in response")

    if len(coords) < 3:
        print(f"[WARNING] Insufficient vertices ({len(coords)}) for osm_id={osm_id}")
        _osm_cache[osm_id] = None
        return None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    _osm_cache[osm_id] = coords
    return coords


def build_nofly_polygons(config: dict) -> tuple[list, list]:
    if not config.get("nofly_zones", {}).get("enabled", False):
        return [], []
    polygons = []
    meta = []
    for zone in config["nofly_zones"]["zones"]:
        zone_type = zone.get("type", "circle")
        name = zone.get("name", "未命名")

        if zone_type == "osm_way":
            coords = fetch_airport_polygon(zone["osm_way_id"])
            if coords is not None:
                polygons.append(Polygon(coords))
                meta.append({"name": name, "is_real": True, "type": "osm_way", "coords": coords})
                print(f"  [禁飞区] {name}: OSM真实边界 ({len(coords)} 顶点)")
            else:
                print(f"  [禁飞区] {name}: OSM获取失败，回退至圆形")
                center = Point(zone["center_lon"], zone["center_lat"])
                radius_deg = zone["radius_km"] / 111.0
                polygons.append(center.buffer(radius_deg))
                meta.append({"name": name, "is_real": False, "type": "circle_fallback",
                             "center_lat": zone["center_lat"], "center_lon": zone["center_lon"],
                             "radius_km": zone["radius_km"]})

        elif zone_type == "circle":
            center = Point(zone["center_lon"], zone["center_lat"])
            radius_deg = zone["radius_km"] / 111.0
            polygons.append(center.buffer(radius_deg))
            meta.append({"name": name, "is_real": False, "type": "circle",
                         "center_lat": zone["center_lat"], "center_lon": zone["center_lon"],
                         "radius_km": zone["radius_km"]})
            print(f"  [禁飞区] {name}: 圆形近似 (半径 {zone['radius_km']} km)")

        elif zone_type == "polygon":
            coords = zone["coordinates"]
            polygons.append(Polygon(coords))
            meta.append({"name": name, "is_real": True, "type": "polygon", "coords": coords})
            print(f"  [禁飞区] {name}: 手动多边形 ({len(coords)} 顶点)")

    return polygons, meta


def edge_intersects_nofly(lon1, lat1, lon2, lat2, nofly_polygons: list) -> bool:
    if not nofly_polygons:
        return False
    line = LineString([(lon1, lat1), (lon2, lat2)])
    for poly in nofly_polygons:
        if line.intersects(poly):
            return True
    return False


def point_in_nofly(lon, lat, nofly_polygons: list) -> bool:
    if not nofly_polygons:
        return False
    pt = Point(lon, lat)
    for poly in nofly_polygons:
        if pt.within(poly):
            return True
    return False


if __name__ == "__main__":
    distance = haversine_distance(117.346, 39.125, 117.197, 39.134)
    print(f"天津滨海机场到天津站距离：{distance / 1000:.2f} 公里")
    # Test nofly functions
    test_config = {
        "nofly_zones": {
            "enabled": True,
            "zones": [
                {
                    "name": "天津滨海国际机场禁飞区",
                    "type": "osm_way",
                    "osm_way_id": 107015361,
                    "fallback_type": "circle",
                    "center_lat": 39.1244,
                    "center_lon": 117.3460,
                    "radius_km": 5.0,
                }
            ],
        }
    }
    polys, meta = build_nofly_polygons(test_config)
    print(f"禁飞区多边形数: {len(polys)}")
    if polys:
        poly = polys[0]
        xy = poly.exterior.xy
        lons, lats = list(xy[0]), list(xy[1])
        print(f"多边形顶点数: {len(lons)}")
        print(f"经度范围: [{min(lons):.4f}, {max(lons):.4f}]")
        print(f"纬度范围: [{min(lats):.4f}, {max(lats):.4f}]")
    print(f"机场坐标在禁飞区内: {point_in_nofly(117.346, 39.1244, polys)}")
    print(f"穿越禁飞区的边: {edge_intersects_nofly(117.2, 39.0, 117.5, 39.2, polys)}")
