import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit_folium import st_folium

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.visualization import plot_network
from src.damage_assessment import load_network, assess_damage
from src.nsga2_recovery import (
    build_recovery_candidates,
    RecoveryProblem,
    run_nsga2,
    extract_pareto_solutions,
    recommend_recovery,
    _compute_base_state,
)


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="低空物流航路网系统",
    page_icon="🚁",
    layout="wide",
)


# ── Cached data loaders ────────────────────────────────────────────────
@st.cache_data
def load_config():
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_network_light():
    """Load network topology from CSVs only (fast, no predictions parquet)."""
    import networkx as nx

    net_dir = _PROJECT_ROOT / "outputs" / "network"
    centers = pd.read_csv(net_dir / "main_centers.csv")
    trunk_edges = pd.read_csv(net_dir / "trunk_edges.csv")
    branch_edges = pd.read_csv(net_dir / "branch_edges.csv")
    trunk_edges["edge_id"] = trunk_edges.apply(
        lambda r: f"{int(r['center_i'])}-{int(r['center_j'])}", axis=1
    )
    G = nx.Graph()
    for _, row in trunk_edges.iterrows():
        G.add_edge(
            int(row["center_i"]), int(row["center_j"]),
            edge_id=row["edge_id"],
            F_norm_ij=row["F_norm_ij"],
            is_redundant=row["is_redundant"],
        )
    return {
        "centers": centers,
        "trunk_edges": trunk_edges,
        "branch_edges": branch_edges,
        "G": G,
    }


@st.cache_data
def load_network_full():
    """Full network load including predictions (for damage assessment pages)."""
    return load_network()


@st.cache_data
def load_poi_clean():
    poi_path = _PROJECT_ROOT / "outputs" / "poi" / "poi_clean.csv"
    return pd.read_csv(poi_path, encoding="utf-8-sig")


# ── Helpers ────────────────────────────────────────────────────────────
def resilience_color(val):
    if val >= 0.8:
        return "normal"
    elif val >= 0.5:
        return "off"
    else:
        return "inverse"


# ── Sidebar navigation ─────────────────────────────────────────────────
st.sidebar.title("低空物流航路网")
page = st.sidebar.radio(
    "导航",
    ["航网概览", "气象风险分析", "受损评估与恢复优化", "关于"],
    label_visibility="collapsed",
)

config = load_config()
districts = config["project"]["districts"]
district_options = ["全部"] + districts

# =====================================================================
# PAGE 1: 航网概览
# =====================================================================
if page == "航网概览":
    st.title("航网概览")

    # Sidebar controls
    st.sidebar.subheader("图层控制")
    show_trunk = st.sidebar.checkbox("主干航路", value=True)
    show_branch = st.sidebar.checkbox("支线", value=True)
    show_centers = st.sidebar.checkbox("主中心", value=True)
    show_subcenters = st.sidebar.checkbox("次中心", value=True)
    show_core = st.sidebar.checkbox("核心需求点", value=False)
    show_aux = st.sidebar.checkbox("辅助需求点", value=False)
    selected_district = st.sidebar.selectbox("按区筛选", district_options)

    # Main layout
    col_left, col_right = st.columns([7, 3])

    with col_left:
        hl = None if selected_district == "全部" else selected_district
        m = plot_network(
            config,
            show_core_poi=show_core,
            show_auxiliary_poi=show_aux,
            highlight_district=hl,
            show_trunk=show_trunk,
            show_branch=show_branch,
            show_centers=show_centers,
            show_subcenters=show_subcenters,
        )
        st_folium(m, width=700, height=550)

    with col_right:
        st.subheader("统计指标")
        network = load_network_light()
        poi = load_poi_clean()
        n_centers = len(network["centers"])
        n_branches = len(network["branch_edges"])
        n_core = int(poi["is_core"].sum())
        center_poi_ids = set(network["centers"]["poi_id"].values)
        core_non_center = poi[poi["is_core"] & ~poi["poi_id"].isin(center_poi_ids)]
        covered = n_core - len(core_non_center)  # centers themselves are core POIs
        coverage_pct = covered / n_core * 100 if n_core > 0 else 0
        pending_path = _PROJECT_ROOT / "outputs" / "network" / "pending_points.csv"
        n_pending = len(pd.read_csv(pending_path)) if pending_path.exists() else 0

        st.metric("主中心", n_centers)
        st.metric("支线总数", n_branches)
        st.metric("核心需求点", n_core)
        st.metric("覆盖率", f"{coverage_pct:.2f}%")
        st.metric("待处理点", n_pending)

# =====================================================================
# PAGE 2: 气象风险分析
# =====================================================================
elif page == "气象风险分析":
    st.title("气象风险分析")

    # Sidebar controls
    st.sidebar.subheader("气象池选择")
    weather_pool = st.sidebar.radio(
        "选择气象池",
        ["常规", "极端风", "极端雨", "风雨复合"],
        horizontal=False,
    )
    fail_threshold = st.sidebar.slider(
        "fail_prob 阈值", 0.1, 0.9, 0.25, 0.05
    )
    affected_threshold = st.sidebar.slider(
        "affected_prob 阈值", 0.05, 0.5, 0.10, 0.05
    )

    # Load data and assess
    network = load_network_full()
    # Temporarily adjust thresholds in config for assessment
    config_patched = dict(config)
    config_patched.setdefault("damage_assessment", {})
    config_patched["damage_assessment"]["fail_prob_threshold"] = fail_threshold
    config_patched["damage_assessment"]["affected_prob_threshold"] = affected_threshold

    damage = assess_damage(network, weather_pool, config_patched)

    # Baseline (常规 pool) for delta comparison
    damage_baseline = None
    if weather_pool != "常规":
        try:
            damage_baseline = assess_damage(network, "常规", config)
        except Exception:
            damage_baseline = None

    # ── Top metrics row ──
    c1, c2, c3 = st.columns(3)

    def _metric_delta(current, baseline):
        if baseline is not None:
            return round(current - baseline, 4)
        return None

    for col, key, label in [
        (c1, "R_struct", "R_struct (结构韧性)"),
        (c2, "R_flow", "R_flow (流量韧性)"),
        (c3, "R_combined", "R_combined (综合韧性)"),
    ]:
        val = damage.get(key, 0)
        delta = _metric_delta(val, damage_baseline.get(key) if damage_baseline else None)
        rc = resilience_color(val)
        with col:
            st.metric(
                label,
                f"{val:.4f}",
                delta=f"{delta:+.4f}" if delta is not None else None,
                delta_color=rc,
            )

    # ── Middle section: map + chart ──
    st.markdown("---")
    col_map, col_chart = st.columns([5, 4])

    with col_map:
        st.subheader("航网受损地图")
        m = plot_network(config, damage_result=damage)
        st_folium(m, width=550, height=450)

    with col_chart:
        st.subheader("主干边失效概率")
        trunk_status = damage["trunk_edge_status"]
        trunk_status = trunk_status.copy()
        trunk_status["color"] = trunk_status["damage_status"].map({
            "failed": "红色(失效)", "affected": "橙色(受影响)", "normal": "蓝色(正常)"
        })
        color_map = {
            "红色(失效)": "red", "橙色(受影响)": "orange", "蓝色(正常)": "blue"
        }
        fig = px.bar(
            trunk_status,
            x="edge_id",
            y="fail_prob",
            color="color",
            color_discrete_map=color_map,
            labels={"fail_prob": "失效概率", "edge_id": "主干边", "color": "状态"},
            height=300,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("支线失效统计")
        n_branch_failed = damage.get("n_branch_failed", 0)
        n_isolated = damage.get("n_isolated_demand", 0)
        n_branch_total = len(network["branch_edges"])
        bc1, bc2 = st.columns(2)
        bc1.metric("失效支线数", n_branch_failed)
        bc2.metric("孤立需求点数", n_isolated)
        st.caption(f"支线总数: {n_branch_total}")

    # ── Bottom: expandable data table ──
    with st.expander("查看边详细数据"):
        display_df = trunk_status[["edge_id", "center_i", "center_j",
                                   "fail_prob", "affected_prob", "damage_status"]]
        st.dataframe(display_df, use_container_width=True)

# =====================================================================
# PAGE 3: 受损评估与恢复优化
# =====================================================================
elif page == "受损评估与恢复优化":
    st.title("受损评估与恢复优化")

    # Sidebar controls
    st.sidebar.subheader("优化设置")
    recovery_pool = st.sidebar.radio(
        "选择气象池",
        ["极端风", "极端雨", "风雨复合"],
        horizontal=False,
    )
    budget_factor = st.sidebar.slider(
        "预算系数",
        0.2, 1.0, 0.5, 0.05,
        help="1.0 = 选择所有候选方案",
    )

    run_btn = st.sidebar.button("运行 NSGA-II 优化", type="primary", use_container_width=True)

    # Initialize session state
    if "nsga2_cache" not in st.session_state:
        st.session_state.nsga2_cache = {}  # keyed by (pool, budget_factor)

    cache_key = (recovery_pool, budget_factor)

    # Main area
    if run_btn or cache_key in st.session_state.nsga2_cache:
        with st.spinner("正在运行损伤评估与 NSGA-II 优化..."):
            if cache_key not in st.session_state.nsga2_cache:
                network = load_network_full()
                damage = assess_damage(network, recovery_pool, config)
                trunk_edges = network["trunk_edges"]

                candidates = build_recovery_candidates(network, damage)

                base_normal_edges, F_normal_base, F_all, n_isolated_base, n_total_branch = \
                    _compute_base_state(damage, trunk_edges)
                n_total_trunk = len(trunk_edges)

                n_trunk_cand = sum(1 for c in candidates if c["type"] != "branch_reactivate")
                n_branch_cand = sum(1 for c in candidates if c["type"] == "branch_reactivate")
                avg_trunk = np.mean(
                    [c["cost"] for c in candidates if c["type"] != "branch_reactivate"] or [1.0]
                )
                max_budget = n_trunk_cand * avg_trunk + n_branch_cand * 0.5
                budget = max_budget * budget_factor

                problem = RecoveryProblem(
                    candidates, base_normal_edges, F_normal_base, F_all,
                    n_total_trunk, n_isolated_base, n_total_branch, budget
                )
                result = run_nsga2(problem, config)

                pareto_df = extract_pareto_solutions(result, candidates, damage, trunk_edges)
                recommended = recommend_recovery(pareto_df)

                # Parse recovery edges from recommended solution
                rec_edges_raw = recommended.get("selected_edges", "")
                if isinstance(rec_edges_raw, str):
                    recovery_edge_list = [e.strip() for e in rec_edges_raw.split(",") if e.strip()]
                else:
                    recovery_edge_list = []

                st.session_state.nsga2_cache[cache_key] = {
                    "damage": damage,
                    "pareto_df": pareto_df,
                    "recommended": recommended,
                    "recovery_edge_list": recovery_edge_list,
                }
            else:
                cached = st.session_state.nsga2_cache[cache_key]
                damage = cached["damage"]
                pareto_df = cached["pareto_df"]
                recommended = cached["recommended"]
                recovery_edge_list = cached["recovery_edge_list"]
                network = load_network_light()

        # ── Top metrics ──
        st.subheader("恢复效果对比")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("优化前 R_combined", f"{damage['R_combined']:.4f}")
        c2.metric("优化后 R_combined", f"{recommended['R_combined']:.4f}",
                  delta=f"{recommended['R_combined'] - damage['R_combined']:+.4f}")
        c3.metric("Pareto 解数量", len(pareto_df))
        c4.metric("推荐方案代价", f"{recommended['total_cost']:.2f}")

        # ── Middle section ──
        st.markdown("---")
        col_map, col_chart = st.columns([5, 4])

        with col_map:
            st.subheader("恢复方案地图")
            m = plot_network(
                config,
                damage_result=damage,
                recovery_edges=recovery_edge_list,
            )
            st_folium(m, width=550, height=450)

        with col_chart:
            st.subheader("Pareto 前沿")

            # Pareto scatter: X=R_struct, Y=1-isolation_rate
            pareto_plot = pareto_df.copy()
            pareto_plot["connectivity"] = 1.0 - pareto_plot["isolation_rate"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pareto_plot["R_struct"],
                y=pareto_plot["connectivity"],
                mode="markers",
                name="Pareto 解",
                marker=dict(size=8, color="lightblue", line=dict(width=1, color="gray")),
                hovertemplate="R_struct=%{x:.3f}, 连通=%{y:.3f}<br>R_combined=%{customdata:.3f}",
                customdata=pareto_plot["R_combined"],
            ))
            # Star for recommended
            fig.add_trace(go.Scatter(
                x=[recommended["R_struct"]],
                y=[1.0 - recommended["isolation_rate"]],
                mode="markers+text",
                name="推荐方案",
                marker=dict(size=16, symbol="star", color="gold",
                            line=dict(width=1, color="darkorange")),
                text=["推荐"],
                textposition="top center",
            ))
            fig.update_layout(
                xaxis_title="R_struct (结构韧性)",
                yaxis_title="1 - isolation_rate (连通性)",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("推荐方案详情")
            st.info(
                f"**恢复边总数**: {recommended['n_reactivated'] + recommended['n_detour'] + recommended['n_branch_recovered']} 条\n\n"
                f"- 主干重新激活: {recommended['n_reactivated']} 条\n"
                f"- 绕行边: {recommended['n_detour']} 条\n"
                f"- 支线恢复: {recommended['n_branch_recovered']} 条\n\n"
                f"**总代价**: {recommended['total_cost']:.2f}"
            )

        # ── Bottom: expandable Pareto table ──
        with st.expander("查看所有 Pareto 解"):
            display_cols = ["solution_id", "R_struct", "isolation_rate", "R_combined",
                            "total_cost", "n_reactivated", "n_detour", "n_branch_recovered"]
            available_cols = [c for c in display_cols if c in pareto_df.columns]
            st.dataframe(pareto_df[available_cols], use_container_width=True)
    else:
        st.info("请选择气象池并点击「运行 NSGA-II 优化」按钮开始分析")

# =====================================================================
# PAGE 4: 关于
# =====================================================================
elif page == "关于":
    st.title("关于")

    st.subheader("项目简介")
    st.markdown("""
    本项目实现**城市低空物流分层航路网构建与气象韧性恢复系统**，
    针对天津中心城区-东丽区组合研究区（7区）进行低空物流航路网规划、
    气象风险评估与韧性恢复优化。
    """)

    st.subheader("研究区域")
    st.markdown("天津中心城区 — 东丽区组合研究区（7区：东丽、和平、河北、河东、河西、南开、红桥）")

    st.subheader("算法链条")
    st.markdown("""
    1. **POI 抓取与清洗** — 高德API获取7区物流与生活POI，清洗分类为核心/辅助需求点
    2. **层次聚类** — 基于需求点空间分布生成主中心与次中心
    3. **引力流量估计** — 计算主中心间潜在物流流量
    4. **ACO 主干生成** — 蚁群优化构建主干航路网，含禁飞区避障与冗余边
    5. **支线连接** — 将核心需求点接入主干网络
    6. **气象风险建模** — ERA5 气象数据 + MLP 神经网络预测边级失效概率
    7. **损伤评估** — 按气象池评估航路网结构/流量韧性
    8. **NSGA-II 恢复优化** — 多目标优化求解最优恢复方案
    """)

    st.subheader("关键指标汇总")

    network = load_network_light()
    poi = load_poi_clean()

    n_centers = len(network["centers"])
    n_trunk = len(network["trunk_edges"])
    n_branches = len(network["branch_edges"])
    n_core = int(poi["is_core"].sum())

    # Coverage
    center_poi_ids = set(network["centers"]["poi_id"].values)
    core_non_center = poi[poi["is_core"] & ~poi["poi_id"].isin(center_poi_ids)]
    covered = n_core - len(core_non_center)
    coverage_pct = covered / n_core * 100 if n_core > 0 else 0

    metrics_data = {
        "指标": ["POI 总数", "主中心数", "主干边数", "支线数", "核心需求点数",
                 "覆盖率", "MLP 模型准确率", "主干边数(含冗余)"],
        "数值": [
            len(poi),
            n_centers,
            n_trunk,
            n_branches,
            n_core,
            f"{coverage_pct:.2f}%",
            "99.89%",
            f"{n_trunk} (基干 {n_trunk - int(network['trunk_edges']['is_redundant'].sum())} + 冗余 {int(network['trunk_edges']['is_redundant'].sum())})",
        ],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # Per-pool resilience summary
    st.subheader("各气象池韧性指标 (R_combined)")
    try:
        pool_summary = []
        for pool_name in ["常规", "极端风", "极端雨", "风雨复合"]:
            try:
                dmg = assess_damage(network, pool_name, config)
                pool_summary.append({
                    "气象池": pool_name,
                    "R_struct": f"{dmg['R_struct']:.4f}",
                    "R_flow": f"{dmg['R_flow']:.4f}",
                    "R_combined": f"{dmg['R_combined']:.4f}",
                })
            except Exception:
                pool_summary.append({
                    "气象池": pool_name,
                    "R_struct": "N/A",
                    "R_flow": "N/A",
                    "R_combined": "N/A",
                })
        st.dataframe(pd.DataFrame(pool_summary), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"无法加载韧性指标: {e}")

    st.subheader("数据来源")
    st.markdown("""
    - **POI 数据**: 高德地图 API (天津7区, 7,699条)
    - **气象数据**: ERA5 再分析数据 (2021-2025, 10m风 + 总降水)
    - **气象观测**: METAR ZBTJ (天津滨海国际机场, 2021-2025)
    - **禁飞区**: OpenStreetMap (天津滨海国际机场真实多边形边界)
    - **底图**: OpenStreetMap
    """)

    st.subheader("技术栈")
    st.markdown("""
    Python 3.10+, Streamlit, Folium, scikit-learn, NetworkX, PyMoo (NSGA-II),
    PyTorch, GeoPandas, PyProj, Plotly
    """)


# ── Footer ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("低空物流航路网系统 v1.0")
st.sidebar.caption("阶段1-5 ✅")
