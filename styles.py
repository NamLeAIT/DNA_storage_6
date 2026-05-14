from __future__ import annotations

import streamlit as st


def apply_style() -> None:
    st.markdown(
        """
<style>
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2rem;
    max-width: 1260px;
}
h1 { font-size: 32px !important; font-weight: 800 !important; }
h2 { font-size: 25px !important; font-weight: 760 !important; }
h3 { font-size: 21px !important; font-weight: 720 !important; margin-top: 0.3rem !important; }
p, label, div, span { font-size: 15.5px; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(250,252,255,0.98), rgba(244,248,252,0.98));
    border: 1px solid rgba(82, 115, 150, 0.22) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(90,125,156,0.18);
    border-radius: 12px;
    padding: 0.45rem 0.65rem;
}
div[data-testid="stMetricLabel"] p { font-size: 13px !important; }
div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 760 !important; }
.stButton button {
    border-radius: 10px !important;
    font-weight: 650 !important;
}
.small-note {
    color: #64748b;
    font-size: 13.5px;
    line-height: 1.42;
}
.pipeline-box {
    padding: 0.8rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(90,125,156,0.18);
    background: rgba(255,255,255,0.74);
}
</style>
""",
        unsafe_allow_html=True,
    )
