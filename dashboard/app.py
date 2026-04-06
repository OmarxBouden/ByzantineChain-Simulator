"""Streamlit dashboard for the free-rider simulation.

Reads scenario output dirs under output/ and renders four panels:
  1. Timeline   — round-by-round dynamics
  2. Agent Profiles — per-agent deep dive
  3. Detection Metrics — signals that separate honest from dishonest
  4. Assessor Inspector — placeholder for Phase 2
Plus a comparison mode to overlay two scenarios.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output"

TYPE_COLORS = {
    "honest": "#2ecc71",
    "free_rider": "#e74c3c",
    "strategic": "#f39c12",
    "selective": "#9b59b6",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _ensure_output():
    """Run the baseline scenario if no scenario output exists."""
    if list_scenarios():
        return
    with st.spinner("No output found — running baseline simulation..."):
        subprocess.run(
            [sys.executable, "-m", "engine.simulation", "baseline"],
            cwd=str(ROOT),
            check=True,
        )


def list_scenarios() -> list[str]:
    """Return names of saved scenario directories."""
    if not OUTPUT_ROOT.exists():
        return []
    return sorted(
        d.name for d in OUTPUT_ROOT.iterdir()
        if d.is_dir() and (d / "events.jsonl").exists()
    )


@st.cache_data
def load_scenario(name: str) -> tuple[pd.DataFrame, dict, dict]:
    """Load events, ground truth, and config for a scenario."""
    scenario_dir = OUTPUT_ROOT / name

    rows = []
    with open(scenario_dir / "events.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    with open(scenario_dir / "ground_truth.json") as f:
        gt = json.load(f)

    config_path = scenario_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return df, gt, config


def split_events(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {t: df[df["type"] == t].copy() for t in df["type"].unique()}


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="ByzantineChain", layout="wide")
st.title("ByzantineChain Dashboard")

_ensure_output()
scenarios = list_scenarios()

if not scenarios:
    st.error("No scenarios found. Run: `python -m engine.simulation baseline`")
    st.stop()

# ── Sidebar: scenario selection ──────────────────────────────────────────────

with st.sidebar:
    st.header("Scenario")

    mode = st.radio("Mode", ["Single", "Compare"], horizontal=True)

    if mode == "Single":
        selected = st.selectbox("Select scenario", scenarios)
        df, gt, config = load_scenario(selected)
        tables = split_events(df)
        n_rounds = int(df["round"].max())

        st.markdown(f"**{config.get('name', selected)}**")
        if config.get("description"):
            st.caption(config["description"])
        st.markdown(f"{n_rounds} rounds · {len(gt)} agents")
        st.markdown(
            " / ".join(
                f"**{sum(1 for v in gt.values() if v == t)} {t}**"
                for t in TYPE_COLORS if any(v == t for v in gt.values())
            )
        )
        if config:
            with st.expander("Full config"):
                st.json(config)

    else:  # Compare
        col1, col2 = st.columns(2)
        with col1:
            sel_a = st.selectbox("Scenario A", scenarios, key="cmp_a")
        with col2:
            sel_b = st.selectbox("Scenario B", scenarios,
                                 index=min(1, len(scenarios) - 1), key="cmp_b")
        df_a, gt_a, cfg_a = load_scenario(sel_a)
        df_b, gt_b, cfg_b = load_scenario(sel_b)

    st.markdown("---")
    st.markdown("**Run new scenarios:**")
    st.code("python -m engine.simulation --list\npython -m engine.simulation baseline\npython -m engine.simulation --all", language="bash")


# ── Helpers ──────────────────────────────────────────────────────────────────

def agent_color(agent_id: str, ground_truth: dict) -> str:
    return TYPE_COLORS.get(ground_truth.get(agent_id, ""), "#95a5a6")


def build_summary(df: pd.DataFrame, gt: dict) -> pd.DataFrame:
    """Build per-agent summary table from events."""
    tables = split_events(df)
    settlements = tables.get("settlement", pd.DataFrame())
    results = tables.get("result_submitted", pd.DataFrame())
    assignments = tables.get("agent_assigned", pd.DataFrame())
    verifications = tables.get("verification_outcome", pd.DataFrame())
    bids = tables.get("bid_submitted", pd.DataFrame())

    rows = []
    for aid in sorted(gt.keys()):
        wins = len(assignments[assignments["agent_id"] == aid]) if not assignments.empty else 0

        accept_rate = None
        if not verifications.empty:
            v = verifications[verifications["agent_id"] == aid]
            if len(v) > 0:
                accept_rate = (v["verdict"] == "accept").sum() / len(v)

        avg_bid = bids[bids["agent_id"] == aid]["bid"].mean() if not bids.empty else None

        avg_quality = None
        if not results.empty:
            ar = results[results["agent_id"] == aid]
            if len(ar) > 0:
                avg_quality = ar["quality"].mean()

        balance = 0.0
        if not settlements.empty:
            s = settlements[settlements["agent_id"] == aid]
            if len(s) > 0:
                balance = s["cumulative_balance"].iloc[-1]

        rows.append({
            "agent_id": aid, "type": gt[aid], "wins": wins,
            "accept_rate": accept_rate, "avg_bid": avg_bid,
            "avg_quality": avg_quality, "balance": balance,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE MODE
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Compare":
    st.subheader(f"Comparing: **{sel_a}** vs **{sel_b}**")

    tabs_a, tabs_b = split_events(df_a), split_events(df_b)

    # ── Balance comparison ───────────────────────────────────────────────────
    st.markdown("#### Cumulative balance by agent type")
    col1, col2 = st.columns(2)

    for col, label, tbl, ground_truth in [
        (col1, sel_a, tabs_a, gt_a),
        (col2, sel_b, tabs_b, gt_b),
    ]:
        with col:
            st.markdown(f"**{label}**")
            settlements = tbl.get("settlement", pd.DataFrame())
            if not settlements.empty:
                bal = settlements[["round", "agent_id", "cumulative_balance"]].copy()
                bal["type"] = bal["agent_id"].map(ground_truth)
                fig = px.line(
                    bal, x="round", y="cumulative_balance", color="agent_id",
                    hover_data=["type"],
                    labels={"cumulative_balance": "Balance", "round": "Round"},
                )
                for trace in fig.data:
                    trace.line.color = agent_color(trace.name, ground_truth)
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ── Aggregate stats comparison ───────────────────────────────────────────
    st.markdown("#### Aggregate comparison")

    def type_stats(df_in: pd.DataFrame, gt_in: dict) -> pd.DataFrame:
        tbl = split_events(df_in)
        assignments = tbl.get("agent_assigned", pd.DataFrame())
        verifications = tbl.get("verification_outcome", pd.DataFrame())
        results = tbl.get("result_submitted", pd.DataFrame())
        n_rnd = int(df_in["round"].max())

        rows = []
        for agent_type in TYPE_COLORS:
            aids = [a for a, t in gt_in.items() if t == agent_type]
            if not aids:
                continue

            wins = len(assignments[assignments["agent_id"].isin(aids)]) if not assignments.empty else 0
            win_rate = wins / n_rnd if n_rnd > 0 else 0

            accept_rate = 0.0
            if not verifications.empty:
                v = verifications[verifications["agent_id"].isin(aids)]
                if len(v) > 0:
                    accept_rate = (v["verdict"] == "accept").sum() / len(v)

            avg_q = 0.0
            if not results.empty:
                r = results[results["agent_id"].isin(aids)]
                if len(r) > 0:
                    avg_q = r["quality"].mean()

            rows.append({
                "type": agent_type, "count": len(aids),
                "win_rate": win_rate, "accept_rate": accept_rate,
                "avg_quality": avg_q,
            })
        return pd.DataFrame(rows)

    stats_a = type_stats(df_a, gt_a)
    stats_b = type_stats(df_b, gt_b)

    col1, col2 = st.columns(2)
    for col, label, stats in [(col1, sel_a, stats_a), (col2, sel_b, stats_b)]:
        with col:
            st.markdown(f"**{label}**")
            st.dataframe(
                stats.style.format({
                    "win_rate": "{:.1%}", "accept_rate": "{:.1%}",
                    "avg_quality": "{:.3f}",
                }),
                use_container_width=True,
            )

    # ── Detection metrics side by side ───────────────────────────────────────
    st.markdown("#### Quality distribution comparison")
    col1, col2 = st.columns(2)
    for col, label, tbl, ground_truth in [
        (col1, sel_a, tabs_a, gt_a),
        (col2, sel_b, tabs_b, gt_b),
    ]:
        with col:
            st.markdown(f"**{label}**")
            results = tbl.get("result_submitted", pd.DataFrame())
            if not results.empty:
                qd = results[["agent_id", "quality"]].copy()
                qd["type"] = qd["agent_id"].map(ground_truth)
                fig = px.histogram(
                    qd, x="quality", color="type",
                    color_discrete_map=TYPE_COLORS,
                    barmode="overlay", nbins=40, opacity=0.6,
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODE — tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_timeline, tab_profiles, tab_detection, tab_assessor = st.tabs(
    ["Timeline", "Agent Profiles", "Detection Metrics", "Assessor Inspector"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_timeline:
    st.subheader("Round-by-round dynamics")

    settlements = tables.get("settlement", pd.DataFrame())
    if not settlements.empty:
        st.markdown("#### Cumulative balance over rounds")
        bal = settlements[["round", "agent_id", "cumulative_balance"]].copy()
        bal["type"] = bal["agent_id"].map(gt)

        fig_bal = px.line(
            bal, x="round", y="cumulative_balance",
            color="agent_id", hover_data=["type"],
            labels={"cumulative_balance": "Balance", "round": "Round"},
        )
        for trace in fig_bal.data:
            trace.line.color = agent_color(trace.name, gt)
        fig_bal.update_layout(
            height=420, showlegend=False,
            xaxis_title="Round", yaxis_title="Cumulative balance",
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    results = tables.get("result_submitted", pd.DataFrame())
    if not results.empty:
        st.markdown("#### Submitted result quality")
        res = results[["round", "agent_id", "quality", "effort", "min_effort"]].copy()
        res["type"] = res["agent_id"].map(gt)
        res["effort_ratio"] = res["effort"] / res["min_effort"]

        fig_q = px.scatter(
            res, x="round", y="quality", color="type",
            color_discrete_map=TYPE_COLORS,
            hover_data=["agent_id", "effort_ratio"],
            labels={"quality": "Quality", "round": "Round"},
            opacity=0.7,
        )
        fig_q.add_hline(y=0.5, line_dash="dash", line_color="grey",
                        annotation_text="verification threshold")
        fig_q.update_layout(height=380)
        st.plotly_chart(fig_q, use_container_width=True)

    tasks = tables.get("task_created", pd.DataFrame())
    if not tasks.empty:
        st.markdown("#### Task rewards & difficulty")
        col1, col2 = st.columns(2)
        with col1:
            fig_r = px.scatter(
                tasks, x="round", y="reward", size="difficulty",
                labels={"reward": "Reward", "round": "Round"}, opacity=0.6,
            )
            fig_r.update_layout(height=320)
            st.plotly_chart(fig_r, use_container_width=True)
        with col2:
            fig_d = px.histogram(tasks, x="difficulty", nbins=25,
                                 labels={"difficulty": "Difficulty"})
            fig_d.update_layout(height=320)
            st.plotly_chart(fig_d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. AGENT PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_profiles:
    st.subheader("Per-agent statistics")

    summary = build_summary(df, gt)

    def highlight_type(row):
        color = TYPE_COLORS.get(row["type"], "#ffffff")
        return [f"background-color: {color}22" for _ in row]

    st.dataframe(
        summary.style.apply(highlight_type, axis=1).format({
            "accept_rate": "{:.1%}",
            "avg_bid": "{:.1f}",
            "avg_quality": "{:.3f}",
            "balance": "{:+.2f}",
        }, na_rep="—"),
        use_container_width=True,
        height=500,
    )

    st.markdown("---")
    st.markdown("#### Agent drill-down")
    agent_ids = sorted(gt.keys())
    selected_agent = st.selectbox("Select agent", agent_ids,
                                  format_func=lambda a: f"{a} ({gt[a]})")

    bids = tables.get("bid_submitted", pd.DataFrame())
    if not bids.empty:
        col1, col2 = st.columns(2)
        ab = bids[bids["agent_id"] == selected_agent].copy()

        with col1:
            st.markdown("**Bid distribution**")
            fig_bh = px.histogram(ab, x="bid", nbins=30,
                                  labels={"bid": "Bid amount"})
            fig_bh.update_layout(height=300)
            st.plotly_chart(fig_bh, use_container_width=True)

        with col2:
            if "true_cost" in ab.columns:
                st.markdown("**Bid vs true cost**")
                fig_bc = px.scatter(
                    ab, x="true_cost", y="bid",
                    labels={"true_cost": "True cost", "bid": "Bid"}, opacity=0.6,
                )
                fig_bc.add_trace(go.Scatter(
                    x=[ab["true_cost"].min(), ab["true_cost"].max()],
                    y=[ab["true_cost"].min(), ab["true_cost"].max()],
                    mode="lines", line=dict(dash="dash", color="grey"),
                    name="bid = cost",
                ))
                fig_bc.update_layout(height=300)
                st.plotly_chart(fig_bc, use_container_width=True)

    if not results.empty:
        ar = results[results["agent_id"] == selected_agent]
        if len(ar) > 0:
            st.markdown("**Quality over rounds**")
            fig_aq = px.scatter(ar, x="round", y="quality", opacity=0.7,
                                labels={"quality": "Quality", "round": "Round"})
            fig_aq.add_hline(y=0.5, line_dash="dash", line_color="grey")
            fig_aq.update_layout(height=280)
            st.plotly_chart(fig_aq, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETECTION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_detection:
    st.subheader("Signals for detecting free-riders")
    st.caption("These metrics use ground-truth labels for evaluation — "
               "the Assessor will NOT have access to them.")

    bids = tables.get("bid_submitted", pd.DataFrame())
    results = tables.get("result_submitted", pd.DataFrame())

    if not bids.empty:
        st.markdown("#### Bid / true-cost ratio by agent type")
        bc = bids[["agent_id", "bid", "true_cost"]].copy()
        bc["type"] = bc["agent_id"].map(gt)
        bc["bid_ratio"] = bc["bid"] / bc["true_cost"]

        fig_br = px.box(
            bc, x="type", y="bid_ratio", color="type",
            color_discrete_map=TYPE_COLORS,
            labels={"bid_ratio": "bid / true_cost", "type": "Agent type"},
        )
        fig_br.add_hline(y=1.0, line_dash="dash", line_color="grey",
                         annotation_text="honest line")
        fig_br.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_br, use_container_width=True)

    if not results.empty:
        st.markdown("#### Effort / min_effort ratio by type")
        er = results[["agent_id", "effort", "min_effort"]].copy()
        er["type"] = er["agent_id"].map(gt)
        er["effort_ratio"] = er["effort"] / er["min_effort"]

        fig_er = px.box(
            er, x="type", y="effort_ratio", color="type",
            color_discrete_map=TYPE_COLORS,
            labels={"effort_ratio": "effort / min_effort", "type": "Agent type"},
        )
        fig_er.add_hline(y=1.0, line_dash="dash", line_color="grey")
        fig_er.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_er, use_container_width=True)

        st.markdown("#### Quality distribution by type")
        qd = results[["agent_id", "quality"]].copy()
        qd["type"] = qd["agent_id"].map(gt)

        fig_qd = px.histogram(
            qd, x="quality", color="type",
            color_discrete_map=TYPE_COLORS,
            barmode="overlay", nbins=40, opacity=0.6,
            labels={"quality": "Quality score"},
        )
        fig_qd.add_vline(x=0.5, line_dash="dash", line_color="grey",
                         annotation_text="threshold")
        fig_qd.update_layout(height=380)
        st.plotly_chart(fig_qd, use_container_width=True)

    if not results.empty:
        st.markdown("#### Detection accuracy: quality threshold sweep")
        st.caption("If we label agents with mean quality < threshold as 'dishonest', "
                   "how well does that separate them?")

        aq = results.groupby("agent_id")["quality"].mean().reset_index()
        aq["type"] = aq["agent_id"].map(gt)
        aq["is_dishonest"] = aq["type"].isin(["free_rider", "strategic", "selective"])

        thresholds = [t / 100 for t in range(10, 100, 2)]
        roc_rows = []
        for thr in thresholds:
            predicted = aq["quality"] < thr
            tp = (predicted & aq["is_dishonest"]).sum()
            fp = (predicted & ~aq["is_dishonest"]).sum()
            fn = (~predicted & aq["is_dishonest"]).sum()
            tn = (~predicted & ~aq["is_dishonest"]).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            roc_rows.append({"threshold": thr, "TPR": tpr, "FPR": fpr,
                             "precision": precision})

        roc_df = pd.DataFrame(roc_rows)
        col1, col2 = st.columns(2)
        with col1:
            fig_roc = px.line(roc_df, x="FPR", y="TPR",
                              labels={"FPR": "False Positive Rate",
                                      "TPR": "True Positive Rate"})
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="grey"), name="random",
            ))
            fig_roc.update_layout(height=350, title="ROC curve")
            st.plotly_chart(fig_roc, use_container_width=True)
        with col2:
            fig_pr = px.line(roc_df, x="TPR", y="precision",
                             labels={"TPR": "Recall", "precision": "Precision"})
            fig_pr.update_layout(height=350, title="Precision–Recall")
            st.plotly_chart(fig_pr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ASSESSOR INSPECTOR (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_assessor:
    st.subheader("Assessor Inspector")
    st.info("This panel will be populated in Phase 2 when the LLM Assessor is integrated.")

    st.markdown("""
    **Planned features:**
    - Assessor's trust scores per agent over time
    - Side-by-side: Assessor's ranking vs ground truth
    - Assignment decisions explained (Assessor reasoning traces)
    - Confusion matrix: Assessor labels vs true types
    - (α, β) verification parameter tuning
    """)

    st.markdown("---")
    st.markdown("#### Preview: Assessor-visible event log (no ground truth)")

    hidden_cols = {"true_cost", "agent_true_type", "min_effort"}
    preview_df = df.copy()
    for col in hidden_cols:
        if col in preview_df.columns:
            preview_df = preview_df.drop(columns=[col])

    event_filter = st.multiselect(
        "Filter by event type",
        options=sorted(df["type"].unique()),
        default=["agent_assigned", "result_submitted", "verification_outcome", "settlement"],
    )
    round_range = st.slider("Round range", 1, n_rounds, (1, min(10, n_rounds)))

    filtered = preview_df[
        (preview_df["type"].isin(event_filter)) &
        (preview_df["round"] >= round_range[0]) &
        (preview_df["round"] <= round_range[1])
    ]
    st.dataframe(filtered, use_container_width=True, height=400)
