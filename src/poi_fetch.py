from pathlib import Path
from time import sleep

import pandas as pd
import requests
import yaml


AMAP_PLACE_SEARCH_URL = "https://restapi.amap.com/v3/place/text"
POI_COLUMNS = ["poi_id", "name", "type", "lon", "lat", "district"]

ADCODE_MAP = {
    "东丽区": "120110",
    "和平区": "120101",
    "河北区": "120105",
    "河东区": "120102",
    "河西区": "120103",
    "南开区": "120104",
    "红桥区": "120106",
}

# Study area bounding box
LON_MIN = 116.7
LON_MAX = 118.1
LAT_MIN = 38.5
LAT_MAX = 40.3


def fetch_poi(api_key, adcode, poi_type, page_size=25, max_pages=100) -> list[dict]:
    results = []

    for page in range(1, max_pages + 1):
        params = {
            "keywords": poi_type,
            "city": adcode,
            "city_limit": "true",
            "offset": page_size,
            "page": page,
            "key": api_key,
            "output": "JSON",
        }

        response_data = None
        for attempt in range(3):
            try:
                response = requests.get(AMAP_PLACE_SEARCH_URL, params=params, timeout=10)
                response.raise_for_status()
                response_data = response.json()
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    print(f"警告：抓取 {poi_type} 第 {page} 页失败，已停止。原因：{exc}")
                    return results
                sleep(1)

        count = int(response_data.get("count", 0) or 0)
        pois = response_data.get("pois", [])
        if count == 0 or not pois:
            break

        for poi in pois:
            location = poi.get("location")
            if not location:
                continue
            try:
                lon_text, lat_text = location.split(",")[:2]
                lon = float(lon_text)
                lat = float(lat_text)
            except (ValueError, TypeError):
                continue

            results.append(
                {
                    "poi_id": poi.get("id"),
                    "name": poi.get("name"),
                    "type": poi_type,
                    "lon": lon,
                    "lat": lat,
                }
            )

    return results


def fetch_all_poi(api_key, districts: list[str], poi_types,
                  page_size=25, max_pages=100) -> pd.DataFrame:
    records = []
    for district in districts:
        adcode = ADCODE_MAP[district]
        for poi_type in poi_types:
            pois = fetch_poi(api_key, adcode, poi_type, page_size, max_pages)
            for p in pois:
                p["district"] = district
            records.extend(pois)

    df = pd.DataFrame(records, columns=POI_COLUMNS)

    if not df.empty:
        in_bbox = (
            (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX) &
            (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
        )
        filtered_out = (~in_bbox).sum()
        df = df[in_bbox].reset_index(drop=True)
        print(f"坐标过滤: 保留 {len(df)} 条, 过滤掉 {filtered_out} 条")

        df = df.drop_duplicates(subset=["poi_id"], keep="first").reset_index(drop=True)
        print(f"去重后: {len(df)} 条")

    return df[["poi_id", "name", "type", "lon", "lat", "district"]]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config["amap"]["api_key"]
    districts = config["project"]["districts"]
    page_size = config["amap"].get("page_size", 25)
    max_pages = config["amap"].get("max_pages", 100)
    poi_types = config["poi"]["core_types"] + config["poi"]["auxiliary_types"]

    df = fetch_all_poi(api_key, districts, poi_types, page_size, max_pages)
    print(f"总条数：{len(df)}")
    if not df.empty:
        print("\n各区各类型数量:")
        print(df.groupby(["district", "type"]).size().to_string())
        print(f"\n各区总数:")
        print(df["district"].value_counts().to_string())

    output_path = project_root / "outputs" / "poi" / "poi_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{output_path}")
