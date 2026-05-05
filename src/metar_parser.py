import re
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_metar_path(data_dir="data/raw/metar"):
    x_path = Path("X:/data/raw/metar")
    if x_path.exists():
        return x_path
    rel_path = _PROJECT_ROOT / data_dir
    if rel_path.exists():
        return rel_path
    abs_path = Path(data_dir)
    if abs_path.exists():
        return abs_path
    raise FileNotFoundError(f"找不到 METAR 数据目录: {data_dir}")


_WIND_RE_MPS = re.compile(r"(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?MPS")
_WIND_RE_KT = re.compile(r"(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT")
_TEMP_RE = re.compile(r"(M?\d{2})/(M?\d{2})")


def _parse_wind_mps(metar_str):
    m = _WIND_RE_MPS.search(metar_str)
    if m:
        return float(m.group(2))
    m = _WIND_RE_KT.search(metar_str)
    if m:
        return float(m.group(2)) * 0.5144
    return np.nan


def _parse_temp(metar_str):
    m = _TEMP_RE.search(metar_str)
    if m:
        t_str = m.group(1)
        if t_str.startswith("M"):
            return -float(t_str[1:])
        return float(t_str)
    return np.nan


def _parse_precip(metar_str):
    has_ra = bool(re.search(r"\bRA\b", metar_str))
    has_rasn = "RASN" in metar_str
    has_sn = bool(re.search(r"\bSN\b", metar_str))
    has_precip = has_ra or has_rasn or has_sn
    return has_precip, 1.0 if has_precip else 0.0


def _parse_cavok(metar_str):
    return "CAVOK" in metar_str


def parse_metar_files(data_dir="data/raw/metar") -> pd.DataFrame:
    """
    Parse all ZBTJ_*.txt METAR files from data_dir.

    Extracts wind speed (m/s), precipitation flag, temperature (C), CAVOK flag
    from raw METAR strings.

    Returns DataFrame with columns: valid, wind_mps, precip_mm, has_precip, tmpc, cavok
    """
    metar_path = _resolve_metar_path(data_dir)
    txt_files = sorted(metar_path.glob("ZBTJ_*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No ZBTJ_*.txt files found in {metar_path}")

    print(f"找到 {len(txt_files)} 个 METAR 文件:")
    dfs = []
    for f in txt_files:
        df = pd.read_csv(f)
        df = df[df["station"] != "station"]  # skip any repeated headers
        print(f"  - {f.name}: {len(df)} 条记录")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["valid"] = pd.to_datetime(df_all["valid"])
    df_all = df_all.sort_values("valid").reset_index(drop=True)

    metar_strs = df_all["metar"].astype(str)

    df_all["wind_mps"] = metar_strs.apply(_parse_wind_mps)
    df_all["tmpc"] = metar_strs.apply(_parse_temp)

    precip_results = metar_strs.apply(_parse_precip)
    df_all["has_precip"] = precip_results.apply(lambda x: x[0])
    df_all["precip_mm"] = precip_results.apply(lambda x: x[1])

    df_all["cavok"] = metar_strs.apply(_parse_cavok)

    result = df_all[["valid", "wind_mps", "precip_mm", "has_precip", "tmpc", "cavok"]].copy()

    before = len(result)
    result = result.dropna(subset=["wind_mps"])
    if before != len(result):
        print(f"删除 {before - len(result)} 条风速缺失记录")

    result = result.sort_values("valid").reset_index(drop=True)

    print(f"\nMETAR 解析完成: {len(result)} 条有效记录")
    print(f"时间范围: {result['valid'].min()} ~ {result['valid'].max()}")

    return result


def compare_era5_metar(df_era5, df_metar) -> None:
    """
    Cross-validate ERA5 against METAR observations.

    Prints wind speed and precipitation statistics for comparison.
    """
    print("\n" + "=" * 60)
    print("ERA5 vs METAR 交叉验证")
    print("=" * 60)

    metar_wind_mean = df_metar["wind_mps"].mean()
    metar_wind_p90 = df_metar["wind_mps"].quantile(0.90)
    metar_wind_max = df_metar["wind_mps"].max()

    era5_wind_mean = df_era5["wind_speed"].mean()
    era5_wind_p90 = df_era5["wind_speed"].quantile(0.90)
    era5_wind_max = df_era5["wind_speed"].max()

    print(f"\n风速统计:")
    print(f"  {'':>12} {'均值(m/s)':>12} {'P90(m/s)':>12} {'最大(m/s)':>12}")
    print(f"  {'METAR':>12} {metar_wind_mean:>12.4f} {metar_wind_p90:>12.4f} {metar_wind_max:>12.4f}")
    print(f"  {'ERA5':>12} {era5_wind_mean:>12.4f} {era5_wind_p90:>12.4f} {era5_wind_max:>12.4f}")

    metar_precip_ratio = df_metar["has_precip"].mean()
    era5_rain_mean = df_era5["tp_mm_day"].mean()
    if "weather_pool" in df_era5.columns:
        era5_extreme_rain_ratio = (df_era5["weather_pool"].isin(["极端雨", "风雨复合"])).mean()
    else:
        rain_p90 = df_era5["tp_mm_day"].quantile(0.90)
        era5_extreme_rain_ratio = (df_era5["tp_mm_day"] >= rain_p90).mean()

    print(f"\n降水统计:")
    print(f"  METAR 有降水记录占比: {metar_precip_ratio:.4f} ({metar_precip_ratio*100:.2f}%)")
    print(f"  ERA5 降水均值: {era5_rain_mean:.4f} mm/day")
    print(f"  ERA5 P90 极端降水占比: {era5_extreme_rain_ratio:.4f} ({era5_extreme_rain_ratio*100:.2f}%)")

    wind_ratio = metar_wind_mean / era5_wind_mean if era5_wind_mean > 0 else float("inf")
    consistent = 0.5 < wind_ratio < 2.0
    print(f"\n结论: ", end="")
    if consistent:
        print("交叉验证完成，数量级一致")
    else:
        print(f"交叉验证完成，存在偏差 (METAR/ERA5 风速比值={wind_ratio:.2f})")


if __name__ == "__main__":
    print("=" * 60)
    print("METAR 本地文件解析")
    print("=" * 60)

    df_metar = parse_metar_files()
    print(f"\n总记录数: {len(df_metar)}")
    print(f"时间范围: {df_metar['valid'].min()} ~ {df_metar['valid'].max()}")

    print(f"\n风速统计:")
    print(f"  均值: {df_metar['wind_mps'].mean():.4f} m/s")
    print(f"  P90:  {df_metar['wind_mps'].quantile(0.90):.4f} m/s")
    print(f"  最大值: {df_metar['wind_mps'].max():.4f} m/s")
    print(f"有降水记录占比: {df_metar['has_precip'].mean()*100:.2f}%")
    print(f"CAVOK 占比: {df_metar['cavok'].mean()*100:.2f}%")
    print(f"温度范围: {df_metar['tmpc'].min():.1f}°C ~ {df_metar['tmpc'].max():.1f}°C")

    output_path = _PROJECT_ROOT / "data" / "raw" / "metar_ZBTJ_parsed.csv"
    df_metar.to_csv(output_path, index=False)
    print(f"\n解析结果保存至: {output_path}")

    samples_path = _PROJECT_ROOT / "data" / "processed" / "edge_weather_samples.parquet"
    if samples_path.exists():
        print("\n" + "=" * 60)
        print("与 ERA5 交叉验证")
        print("=" * 60)
        df_edge = pd.read_parquet(samples_path)
        df_era5_agg = df_edge.groupby("time")[["wind_speed", "tp_mm_day"]].mean().reset_index()
        df_era5_agg = df_era5_agg.rename(columns={"wind_speed": "wind_speed", "tp_mm_day": "tp_mm_day"})
        compare_era5_metar(df_era5_agg, df_metar)
    else:
        print(f"\n(跳过 ERA5 交叉验证: {samples_path} 不存在)")
