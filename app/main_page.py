from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Детекция дефектов ЛЭП",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

page = st.navigation(
    [
        st.Page(str(ROOT_DIR / "pages" / "main_page.py"), title="🏠 Главная"),
        st.Page(str(ROOT_DIR / "pages" / "detection_page.py"), title="🔎 Детекция"),
        st.Page(str(ROOT_DIR / "pages" / "history_page.py"), title="📋 История"),
    ]
)

page.run()
