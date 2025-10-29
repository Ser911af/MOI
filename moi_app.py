import os
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# ===================== CONFIG =====================
st.set_page_config(page_title="Frutto – MOI Dashboard", layout="wide")
load_dotenv()


def _get_secret(name: str, default=None):
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_ANON_KEY")
SUPABASE_TABLE = _get_secret("SUPABASE_TABLE", "ventas_frutto")
LOGO_PATH = _get_secret("LOGO_PATH", "Logo/WhatsApp Image 2025-08-26 at 1.50.59 PM (1).jpeg")

# ===================== PASSWORD =====================
APP_PASSWORD = _get_secret("MOI_PASSWORD", None)
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 MOI – Restricted Access")
    if not APP_PASSWORD:
        st.warning("MOI_PASSWORD not found in secrets/.env. Please set a secure password.")
    pwd = st.text_input("Enter password:", type="password")
    if st.button("Enter"):
        if APP_PASSWORD and pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Access granted ✅")
            st.rerun()
        else:
            st.error("❌ Incorrect password")
    st.stop()

# ===================== COLUMN NAMES =====================
COL_DATE = "reqs_date"
COL_REP = "sales_rep"
COL_INV = "invoice_num"
COL_REV = "total_revenue"
COL_PROF = "total_profit_usd"
COL_STATUS = "invoice_payment_status"
NEEDED = [COL_DATE, COL_REP, COL_INV, COL_REV, COL_PROF, COL_STATUS]

# ===================== FORMATTING =====================
def fmt_money(v, decimals: int = 0):
    try:
        return f"${v:,.{decimals}f}"
    except Exception:
        return v


def fmt_pct_unit(v, decimals: int = 1):
    try:
        return f"{(100 * v):.{decimals}f}%"
    except Exception:
        return v


# ===================== MOI PALETTE =====================
PALETA = {
    "Remarkable": "#C00000",
    "Excellent": "#A80E0E",
    "Great": "#E06666",
    "Good": "#4D4D4D",
    "Average": "#7F7F7F",
    "Poor": "#B7B7B7",
}


def _contrast(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 140 else "#000000"


def _badge_html(text: str, bg: str) -> str:
    fg = _contrast(bg)
    return (
        f"<div style='display:block;text-align:center;"
        f"background:{bg};color:{fg};font-weight:700;"
        f"padding:2px 8px;border-radius:10px;line-height:1.15'>{text}</div>"
    )


def _band_badge(name: str) -> str:
    bg = PALETA.get(name, "#E5E5E5")
    return _badge_html(name, bg)


# ===================== MOI SCALES =====================
MOI_SCALES = {
    "Year": [
        ("Remarkable", 2000000, 0.20, 11000, 900, 10000000),
        ("Excellent", 1520000, 0.19, 10000, 780, 8000000),
        ("Great", 1080000, 0.18, 9000, 660, 6000000),
        ("Good", 850000, 0.17, 8000, 600, 5000000),
        ("Average", 640000, 0.16, 7000, 540, 4000000),
        ("Poor", 450000, 0.15, 6000, 480, 3000000),
    ],
    "Month": [
        ("Remarkable", 166667, 0.20, 11000, 75, 833333),
        ("Excellent", 126667, 0.19, 10000, 65
