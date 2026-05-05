"""阶段2：ERA5 气象数据下载脚本。

按半年/季度分批下载 ERA5 小时级数据，保存为 NetCDF 格式。
CDS API 返回 zip 包（含 instant + accum 两个 NC），脚本自动解压并合并。
"""
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr
import yaml

SPLIT_SCHEDULES = {
    "year": {"全年": list(range(1, 13))},
    "half": {"H1": list(range(1, 7)), "H2": list(range(7, 13))},
    "quarter": {
        "Q1": [1, 2, 3], "Q2": [4, 5, 6],
        "Q3": [7, 8, 9], "Q4": [10, 11, 12],
    },
}


def load_config() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_zip(filepath: Path) -> bool:
    with open(filepath, "rb") as fh:
        return fh.read(2) == b"PK"


def _extract_and_merge_zip(zip_path: Path, output_path: Path) -> bool:
    """Extract NC files from a CDS zip archive and merge into a single NetCDF.

    CDS returns a zip with two files: *_instant.nc (wind) and *_accum.nc (precip).
    All NC read/write happens in a temp dir to avoid netCDF4 C-lib issues with
    non-ASCII paths on Windows.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            print(f"  [WARNING] ZIP 中未找到 .nc 文件")
            return False

        tmpdir = Path(tempfile.mkdtemp())
        try:
            zf.extractall(tmpdir)
            datasets = []
            for name in nc_names:
                ds = xr.open_dataset(str(tmpdir / name), engine="netcdf4")
                datasets.append(ds)
            merged = xr.merge(datasets, compat="override")
            tmp_out = tmpdir / "merged.nc"
            merged.to_netcdf(str(tmp_out))
            shutil.move(str(tmp_out), str(output_path))
            print(f"    合并 {len(nc_names)} 个 NC -> {output_path.name}")
            return True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def download_era5_chunk(year: int, months: list[int], label: str,
                        config: dict, output_dir: Path) -> Path | None:
    """Download a chunk of ERA5 hourly data (specific months).

    Handles CDS returning zip archives by auto-extracting the .nc file.
    """
    area = config["weather"]["area"]
    variables = config["weather"]["era5_variables"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"era5_{year}_{label}.nc"

    if output_path.exists() and not _is_zip(output_path):
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {output_path.name} 已存在 ({size_mb:.1f} MB)")
        return output_path

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [area["north"], area["west"], area["south"], area["east"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    n_months = len(months)
    print(f"  请求 {year} {label} ({n_months} 个月, {', '.join(variables)})...")

    dl_path = output_dir / f"era5_{year}_{label}.dl"
    try:
        client = cdsapi.Client()
        client.retrieve("reanalysis-era5-single-levels", request, str(dl_path))
    except Exception as e:
        err_msg = str(e)
        if "too large" in err_msg.lower() or "cost" in err_msg.lower():
            print(f"  [TOO_LARGE] {label} 请求过大，需进一步拆分")
        else:
            print(f"  [FAIL] {year} {label}: {e}")
        if dl_path.exists():
            dl_path.unlink()
        return None

    # CDS returns a zip with instant + accum NC files
    if _is_zip(dl_path):
        print(f"    解压并合并 instant + accum...")
        ok = _extract_and_merge_zip(dl_path, output_path)
        dl_path.unlink()
        if not ok:
            return None
    else:
        dl_path.rename(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def download_era5_year_adaptive(year: int, config: dict, output_dir: Path) -> bool:
    """Download one year of ERA5, auto-splitting if needed.

    Tries year -> half -> quarter, falling back to smaller chunks on cost-limit errors.
    """
    area = config["weather"]["area"]
    print(f"--- {year} 年 (N{area['north']} S{area['south']} W{area['west']} E{area['east']}) ---")

    full_path = output_dir / f"era5_{year}.nc"
    if full_path.exists() and not _is_zip(full_path):
        print(f"  [SKIP] 全年文件已存在")
        return True

    result = download_era5_chunk(year, list(range(1, 13)), "year", config, output_dir)
    if result:
        return True

    print("  回退至半年拆分...")
    time.sleep(1)
    half_ok = True
    for label, months in SPLIT_SCHEDULES["half"].items():
        r = download_era5_chunk(year, months, label, config, output_dir)
        if r is None:
            half_ok = False
            break
        time.sleep(1)
    if half_ok:
        return True

    print("  回退至季度拆分...")
    time.sleep(1)
    all_ok = True
    for label, months in SPLIT_SCHEDULES["quarter"].items():
        r = download_era5_chunk(year, months, label, config, output_dir)
        if r is None:
            all_ok = False
        time.sleep(1)
    return all_ok


def main():
    config = load_config()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "era5"

    years = list(range(2021, 2026))

    print("=" * 50)
    print("ERA5 气象数据下载（自适应拆分）")
    print(f"输出目录: {output_dir}")
    print(f"年份范围: {years[0]}-{years[-1]}")
    print("=" * 50)

    success = []
    failed = []
    for year in years:
        ok = download_era5_year_adaptive(year, config, output_dir)
        if ok:
            success.append(year)
        else:
            failed.append(year)
        time.sleep(2)

    print("=" * 50)
    print(f"下载完成: 成功 {len(success)} 年, 失败 {len(failed)} 年")
    if success:
        print(f"  成功: {success}")
    if failed:
        print(f"  失败: {failed}")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    config = load_config()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "era5"

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        split_mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
        print(f"单年测试: {year} (模式: {split_mode})")

        if split_mode in SPLIT_SCHEDULES:
            for label, months in SPLIT_SCHEDULES[split_mode].items():
                download_era5_chunk(year, months, label, config, output_dir)
                time.sleep(1)
        else:
            download_era5_year_adaptive(year, config, output_dir)
    else:
        main()
