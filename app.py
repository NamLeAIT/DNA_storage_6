from __future__ import annotations

import streamlit as st

from config import APP_TITLE
from styles import apply_style
from panels import (
    render_panel_1_upload,
    render_panel_2_compression,
    render_panel_3_encoding,
    render_panel_4_experiment,
    render_panel_5_decoding,
    render_panel_6_analysis,
)


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
    apply_style()

    st.title("🧬 DNA Storage Pipeline")
    st.markdown(
        """
<div class="pipeline-box">
<b>New pipeline:</b> Upload → Compression → Encoding → DNA strands preparation → Decoding → Analysis.<br>
</div>
""",
        unsafe_allow_html=True,
    )

    render_panel_1_upload()
    render_panel_2_compression()
    render_panel_3_encoding()
    render_panel_4_experiment()
    render_panel_5_decoding()
    render_panel_6_analysis()


if __name__ == "__main__":
    render_app()
