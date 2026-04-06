"""Streamlit dashboard for the free-rider simulation.

Reads output/events.jsonl + output/ground_truth.json and renders four panels:
  1. Timeline   — round-by-round dynamics
  2. Agent Profiles — per-agent deep dive
  3. Detection Metrics — signals that separate honest from dishonest
  4. Assessor Inspector — placeholder for Phase 2
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Data loading ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS = ROOT / "output" / "events.jsonl"
DEFAULT_GT = ROOT / "output" / "ground_truth.json"

TYPE_COLORS = {
    "honest": "#2ecc71",
    "free_rider": "#e74c3c",
    "strategic": "#f39c12",
    "selective": "#9b59b6",
}


@st.cache_data
def load_events(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data
def load_ground_truth(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def split_events(df: pd.DataFrame):
    """Return sub-DataFrames by event type."""
    return {t: df[df["type"] == t].copy() for t in df["type"].unique()}


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Free-Rider Simulator", layout="wide")
st.title("Free-Rider Simulation Dashboard")

# Sidebar: file selection
with st.sidebar:
    st.header("Data source")
    events_path = st.text_input("events.jsonl", value=str(DEFAULT_EVENTS))
    gt_path = st.text_input("ground_truth.json", value=str(DEFAULT_GT))

    if not Path(events_path).exists():
        st.error("events.jsonl not found — run the simulation first.")
        st.stop()

df = load_events(events_path)
gt = load_ground_truth(gt_path)
tables = split_events(df)

n_rounds = int(df["round"].max())

with st.sidebar:
    st.markdown(f"**{n_rounds}** rounds &middot; **{len(gt)}** agents")
    st.markdown("---")
    st.markdown(
        " / ".join(f"**{sum(1 for v in gt.values() if v == t)} {t}**" for t in TYPE_COLORS)
    )

# ── Helpers ──────────────────────────────────────────────────────────────────


def agent_color(agent_id: str) -> str:
    return TYPE_COLORS.get(gt.get(agent_id, ""), "#95a5a6")


# ── Tab layout ───────────────────────────────────────────────────────────────

tab_timeline, tab_profiles, tab_detection, tab_assessor = st.tabs(
    ["Timeline", "Agent Profiles", "Detection Metrics", "Assessor Inspector"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_timeline:
    st.subheader("Round-by-round dynamics")

    # ── Cumulative balance ───────────────────────────────────────────────────
    settlements = tables.get("settlement", pd.DataFrame())
    if not settlements.empty:
        st.markdown("#### Cumulative balance over rounds")
        bal = settlements[["round", "agent_id", "cumulative_balance"]].copy()
        bal["type"] = bal["agent_id"].map(gt)

        fig_bal = px.line(
            bal, x="round", y="cumulative_balance",
            color="agent_id",
            hover_data=["type"],
            labels={"cumulative_balance": "Balance", "round": "Round"},
        )
        # Color traces by agent type
        for trace in fig_bal.data:
            aid = trace.name
            trace.line.color = agent_color(aid)
        fig_bal.update_layout(
            height=420, showlegend=False,
            xaxis_title="Round", yaxis_title="Cumulative balance",
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    # ── Result quality per round ─────────────────────────────────────────────
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

    # ── Task reward distribution ─────────────────────────────────────────────
    tasks = tables.get("task_created", pd.DataFrame())
    if not tasks.empty:
        st.markdown("#### Task rewards & difficulty")
        col1, col2 = st.columns(2)
        with col1:
            fig_r = px.scatter(
                tasks, x="round", y="reward", size="difficulty",
                labels={"reward": "Reward", "round": "Round"},
                opacity=0.6,
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

    # Build summary table
    assignments = tables.get("agent_assigned", pd.DataFrame())
    verifications = tables.get("verification_outcome", pd.DataFrame())
    bids = tables.get("bid_submitted", pd.DataFrame())

    agent_ids = sorted(gt.keys())
    rows = []
    for aid in agent_ids:
        wins = len(assignments[assignments["agent_id"] == aid]) if not assignments.empty else 0
        if not verifications.empty:
            v = verifications[verifications["agent_id"] == aid]
            accepts = (v["verdict"] == "accept").sum() if len(v) > 0 else 0
            accept_rate = accepts / len(v) if len(v) > 0 else None
        else:
            accepts = 0
            accept_rate = None

        if not bids.empty:
            ab = bids[bids["agent_id"] == aid]
            avg_bid = ab["bid"].mean()
        else:
            avg_bid = None

        if not results.empty:
            ar = results[results["agent_id"] == aid]
            avg_quality = ar["quality"].mean() if len(ar) > 0 else None
        else:
            avg_quality = None

        if not settlements.empty:
            s = settlements[settlements["agent_id"] == aid]
            balance = s["cumulative_balance"].iloc[-1] if len(s) > 0 else 0.0
        else:
            balance = 0.0

        rows.append({
            "agent_id": aid,
            "type": gt[aid],
            "wins": wins,
            "accept_rate": accept_rate,
            "avg_bid": avg_bid,
            "avg_quality": avg_quality,
            "balance": balance,
        })

    summary = pd.DataFrame(rows)

    # Color-coded table
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

    # ── Agent drill-down ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Agent drill-down")
    selected = st.selectbox("Select agent", agent_ids,
                            format_func=lambda a: f"{a} ({gt[a]})")

    if not bids.empty:
        col1, col2 = st.columns(2)
        ab = bids[bids["agent_id"] == selected].copy()

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
                    labels={"true_cost": "True cost", "bid": "Bid"},
                    opacity=0.6,
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
        ar = results[results["agent_id"] == selected]
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

    if not bids.empty:
        # ── Bid / true-cost ratio ────────────────────────────────────────────
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
        # ── Effort ratio ─────────────────────────────────────────────────────
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

        # ── Quality distribution ─────────────────────────────────────────────
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

    # ── Quality-based detection ROC ──────────────────────────────────────────
    if not results.empty:
        st.markdown("#### Detection accuracy: quality threshold sweep")
        st.caption("If we label agents with mean quality < threshold as 'dishonest', "
                   "how well does that separate them?")

        # Per-agent mean quality
        aq = results.groupby("agent_id")["quality"].mean().reset_index()
        aq["type"] = aq["agent_id"].map(gt)
        aq["is_dishonest"] = aq["type"].isin(["free_rider", "strategic", "selective"])

        thresholds = [t / 100 for t in range(10, 100, 2)]
        roc_rows = []
        for thr in thresholds:
            predicted_dishonest = aq["quality"] < thr
            tp = (predicted_dishonest & aq["is_dishonest"]).sum()
            fp = (predicted_dishonest & ~aq["is_dishonest"]).sum()
            fn = (~predicted_dishonest & aq["is_dishonest"]).sum()
            tn = (~predicted_dishonest & ~aq["is_dishonest"]).sum()

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

    # Show what the Assessor will see (no ground-truth fields)
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
