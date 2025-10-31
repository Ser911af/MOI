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
        ("Excellent", 126667, 0.19, 10000, 65, 666667),
        ("Great", 90000, 0.18, 9000, 55, 500000),
        ("Good", 70833, 0.17, 8000, 50, 416667),
        ("Average", 53333, 0.16, 7000, 45, 333333),
        ("Poor", 37500, 0.15, 6000, 40, 250000),
    ],
    "Week": [
        ("Remarkable", 38462, 0.20, 11000, 34, 192308),
        ("Excellent", 29231, 0.19, 10000, 28, 153846),
        ("Great", 20769, 0.18, 9000, 23, 115385),
        ("Good", 16346, 0.17, 8000, 20, 96154),
        ("Average", 12308, 0.16, 7000, 17, 76923),
        ("Poor", 8654, 0.15, 6000, 14, 57692),
    ],
    "Day": [
        ("Remarkable", 5479, 0.20, 11000, 7, 27397),
        ("Excellent", 4164, 0.19, 10000, 6, 21918),
        ("Great", 2959, 0.18, 9000, 5, 16438),
        ("Good", 2329, 0.17, 8000, 4, 13699),
        ("Average", 1753, 0.16, 7000, 4, 10959),
        ("Poor", 1233, 0.15, 6000, 3, 8219),
    ],
}

def _scale_for_range(base_scale, factor: float):
    return [
        (name, p * factor, pct, aov, ords * factor, rev * factor)
        for (name, p, pct, aov, ords, rev) in base_scale
    ]

def _band_helpers(scale):
    order = [t[0] for t in scale]
    by_metric = {
        "profit": {n: p for (n, p, _pct, _aov, _ord, _rev) in scale},
        "pct": {n: pct for (n, _p, pct, _aov, _ord, _rev) in scale},
        "aov": {n: aov for (n, _p, _pct, aov, _ord, _rev) in scale},
        "orders": {n: o for (n, _p, _pct, _aov, o, _rev) in scale},
        "revenue": {n: r for (n, _p, _pct, _aov, _ord, r) in scale},
    }
    return order, by_metric

def _band_for(value, metric, order, by_metric):
    th = by_metric[metric]
    for name in order:
        if value >= th[name]:
            return name
    return order[-1]

def class_moi(row, scale, majority_required: int = 3, reinforce_revenue: bool = True):
    for name, min_profit, min_pct, min_aov, min_orders, min_rev in scale:
        checks = 0
        checks += row["profit_sum"] >= min_profit
        checks += row["profit_pct"] >= min_pct
        checks += row["aov"] >= min_aov
        checks += row["orders"] >= min_orders
        if reinforce_revenue:
            checks += row["revenue_sum"] >= min_rev
        if checks >= majority_required + 1:
            return name
        else:
            if checks >= majority_required:
                return name
    return scale[-1][0]

def count_calendar_periods(dstart, dend, gran):
    s = pd.Timestamp(dstart)
    e = pd.Timestamp(dend)
    if gran == "Year":
        return max(len(pd.period_range(s, e, freq="Y")), 1)
    if gran == "Month":
        return max(len(pd.period_range(s, e, freq="M")), 1)
    if gran == "Week":
        return max(len(pd.period_range(s, e, freq="W-MON")), 1)
    days = pd.date_range(s, e, freq="D")
    return max(int((days.weekday != 6).sum()), 1)

def count_observed_periods(df, gran, col_date):
    if df.empty:
        return 1
    x = df[df[col_date].dt.weekday != 6]
    if gran == "Year":
        return max(x[col_date].dt.to_period("Y").nunique(), 1)
    if gran == "Month":
        return max(x[col_date].dt.to_period("M").nunique(), 1)
    if gran == "Week":
        return max(x[col_date].dt.to_period("W-MON").nunique(), 1)
    return max((x[col_date].dt.floor("D").nunique()), 1)

# ===================== FETCH DATA =====================
def effective_range(d: date, granularity: str, week_mode: str = "window"):
    ts = pd.Timestamp(d)
    if granularity == "Day":
        return ts.date(), ts.date()
    if granularity == "Week":
        if week_mode == "window":
            dstart = ts.date()
            dend = (ts + pd.Timedelta(days=6)).date()
        else:
            dstart = (ts - pd.offsets.Week(weekday=0)).date()
            dend = (pd.Timestamp(dstart) + pd.Timedelta(days=6)).date()
        return dstart, dend
    if granularity == "Month":
        p = ts.to_period("M")
        return p.start_time.date(), p.end_time.date()
    if granularity == "Year":
        p = ts.to_period("Y")
        return p.start_time.date(), p.end_time.date()
    return ts.date(), ts.date()

@st.cache_data(show_spinner=False, ttl=60)
def fetch_server_filtered_v2(dstart: date, dend: date) -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame(columns=NEEDED)

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    start_s = pd.Timestamp(dstart).date().isoformat()
    end_exclusive = (pd.Timestamp(dend) + pd.Timedelta(days=1)).date().isoformat()

    frames: list[pd.DataFrame] = []
    PAGE = 1000
    select_list = ",".join(NEEDED)

    base = (
        client.table(SUPABASE_TABLE)
        .select(select_list)
        .gte(COL_DATE, start_s)
        .lt(COL_DATE, end_exclusive)
        .order(COL_DATE, desc=False)
    )

    start = 0
    while True:
        resp = base.range(start, start+PAGE-1).execute()
        batch = pd.DataFrame(resp.data or [])
        if batch.empty:
            break
        frames.append(batch)
        if len(batch) < PAGE:
            break
        start += PAGE

    if not frames:
        return pd.DataFrame(columns=NEEDED)

    df = pd.concat(frames, ignore_index=True)
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", utc=False)
    df[COL_REV] = pd.to_numeric(df[COL_REV], errors="coerce")
    df[COL_PROF] = pd.to_numeric(df[COL_PROF], errors="coerce")
    if COL_REP in df.columns:
        df[COL_REP] = df[COL_REP].astype("string").str.strip().fillna("(No rep)")
    return df[NEEDED].dropna(subset=[COL_DATE]).reset_index(drop=True)

# ===================== APPLY FILTERS =====================
def apply_filters(df: pd.DataFrame, exclude_sundays: bool, paid_only: bool, exclude_neg: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=NEEDED)

    if not np.issubdtype(df[COL_DATE].dtype, np.datetime64):
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    x = df.copy()
    if exclude_sundays:
        x = x[x[COL_DATE].dt.weekday != 6]
    if paid_only:
        x = x[x[COL_STATUS] == "Paid"]
    if exclude_neg:
        x = x[(x[COL_REV] > 0) & (x[COL_PROF] >= 0)]
    return x

# ===================== NEW: ACTIVITY HELPERS =====================
@st.cache_data(show_spinner=False, ttl=60)
def fetch_daily_activity(dstart: date, dend: date, usernames: list[str] | None) -> pd.DataFrame:
    """
    Actividad diaria (Reached/Engaged/Closed) agregada por fecha.
    Opcional: filtrar por uno o más usernames (daily_metrics.user_name).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame(columns=["d", "reached", "engaged", "closed"])

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Ventana [dstart, dend] en America/Bogota → UTC para consultar created_at
    start_ts = pd.Timestamp(dstart).tz_localize("America/Bogota").tz_convert("UTC")
    end_ts   = (pd.Timestamp(dend) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("America/Bogota").tz_convert("UTC")

    frames: list[pd.DataFrame] = []
    PAGE = 1000
    base = (
        client.table("daily_metrics")
        .select("created_at,user_name,clients_reached_out,clients_engaged,clients_closed")
        .gte("created_at", start_ts.isoformat())
        .lte("created_at", end_ts.isoformat())
        .order("created_at", desc=False)
    )
    if usernames:
        base = base.in_("user_name", usernames)

    start = 0
    while True:
        resp = base.range(start, start+PAGE-1).execute()
        batch = pd.DataFrame(resp.data or [])
        if batch.empty:
            break
        frames.append(batch)
        if len(batch) < PAGE:
            break
        start += PAGE

    if not frames:
        return pd.DataFrame(columns=["d", "reached", "engaged", "closed"])

    df = pd.concat(frames, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce").dt.tz_convert("America/Bogota")
    df["d"] = df["created_at"].dt.tz_localize(None).dt.date

    # Numerificación + guardas lógicas en lectura (no alteramos DB)
    for c in ["clients_reached_out","clients_engaged","clients_closed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["clients_engaged"] = df[["clients_engaged","clients_reached_out"]].min(axis=1)
    df["clients_closed"]  = df[["clients_closed","clients_reached_out"]].min(axis=1)

    g = (df.groupby("d", dropna=False)
            .agg(reached=("clients_reached_out","sum"),
                 engaged=("clients_engaged","sum"),
                 closed =("clients_closed","sum"))
            .reset_index())
    return g

@st.cache_data(show_spinner=False, ttl=300)
def fetch_user_map() -> pd.DataFrame:
    """Mapa username ↔ full_name para alinear ventas (sales_rep) con actividad (user_name)."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame(columns=["username","full_name"])
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = client.table("user_profiles").select("username,full_name").execute()
    return pd.DataFrame(resp.data or [])

def _safe_div(num, den):
    try:
        num = float(num); den = float(den)
        return (num/den) if den else 0.0
    except Exception:
        return 0.0

# ===================== AGG BY REP =====================
def agg_by_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per rep: revenue, profit, orders, profit% and AOV."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["sales_rep", "revenue_sum", "profit_sum", "orders", "profit_pct", "aov"]
        )

    if not np.issubdtype(df[COL_DATE].dtype, np.datetime64):
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df[COL_REV] = pd.to_numeric(df[COL_REV], errors="coerce")
    df[COL_PROF] = pd.to_numeric(df[COL_PROF], errors="coerce")
    df[COL_REP] = df[COL_REP].astype("string").str.strip().fillna("(No rep)")

    g = (
        df.groupby(COL_REP, dropna=False)
          .agg(
              revenue_sum=(COL_REV, "sum"),
              profit_sum=(COL_PROF, "sum"),
              orders=(COL_INV, pd.Series.nunique),
          )
          .reset_index()
    )

    g["profit_pct"] = np.where(
        g["revenue_sum"] > 0, g["profit_sum"] / g["revenue_sum"], 0.0
    )
    g["aov"] = np.where(
        g["orders"] > 0, g["revenue_sum"] / g["orders"], 0.0
    )

    return g.sort_values(["revenue_sum", "orders"], ascending=[False, False]).reset_index(drop=True)

# ===================== HEADER =====================
c1, c2 = st.columns([1, 6])
with c1:
    logo_file = Path(LOGO_PATH)
    if logo_file.exists():
        try:
            from PIL import Image
            st.image(Image.open(logo_file), use_container_width=True)
        except Exception:
            st.empty()
with c2:
    st.title("MOI – Sales Rep View (server-side date filter)")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚙️ Debug")
    debug_ignore_filters = st.checkbox("Ignore business filters", value=False)

    gran = st.radio("Granularity", ["Day", "Week", "Month", "Year"], horizontal=True)
    week_mode = st.radio("Week mode", ["window", "calendar"], index=0,
                         help="window: 7 days starting from base date · calendar: Monday–Sunday")
    base_date = st.date_input("Base date", value=date.today())

    st.markdown("**Targets scale based on…**")
    targets_mode = st.radio("", ["Calendar periods", "Observed periods"],
                            help="Calendar: × number of calendar periods in range · Observed: × periods with data")

    st.markdown("**Business filters (apply to period):**")
    exclude_sundays = st.checkbox("Exclude Sundays", value=True)
    paid_only = st.checkbox("Only 'Paid' invoices", value=False)
    exclude_neg = st.checkbox("Exclude negatives (revenue ≤ 0 or profit < 0)", value=True)

    # === GOAL / WATERLINE ===
    st.markdown("### 🎯 Goal (remaining)")
    goal_remaining = st.number_input(
        "Remaining revenue goal",
        min_value=0,
        value=25_000_000,  # Updated: $25M
        step=50_000,
        help="Revenue still needed within the selected horizon."
    )

    default_goal_end = date(base_date.year, 12, 31)
    goal_end = st.date_input(
        "Goal horizon (until)", value=default_goal_end,
        help="The goal is spread from base date up to this date."
    )

    # NUEVO: usar solo días laborales Mon–Fri para repartir la meta
    workdays_mon_fri = st.checkbox(
        "Workdays only (Mon–Fri)",
        value=True,
        help="Usa solo lunes a viernes para repartir la meta (ignora sábados y domingos)."
    )

    if st.button("Refresh"):
        st.rerun()

# ===================== RANGE =====================
dstart_eff, dend_eff = effective_range(base_date, gran, week_mode)
st.caption(f"**Effective range:** {dstart_eff} → {dend_eff}")

# ===================== FETCH & FILTERS =====================
with st.spinner("Querying Supabase…"):
    df_raw = fetch_server_filtered_v2(dstart_eff, dend_eff)
    df_f = apply_filters(df_raw, exclude_sundays, paid_only, exclude_neg)

with st.expander("🔎 Diagnostics", expanded=False):
    st.write(
        {
            "granularity": gran,
            "week_mode": week_mode,
            "targets_mode": targets_mode,
            "from": str(dstart_eff),
            "to": str(dend_eff),
            "rows_raw": int(len(df_raw)),
            "rows_after_filters": int(len(df_f)),
            "unique_reps": int(df_f[COL_REP].nunique()) if not df_f.empty else 0,
            "min_req": df_f[COL_DATE].min().strftime("%Y-%m-%d") if not df_f.empty else None,
            "max_req": df_f[COL_DATE].max().strftime("%Y-%m-%d") if not df_f.empty else None,
        }
    )
    if not df_f.empty:
        monthly_diag = (
            df_f.assign(month=df_f[COL_DATE].dt.to_period("M").dt.start_time)
            .groupby("month").size().reset_index(name="rows")
        )
        monthly_diag["month"] = monthly_diag["month"].dt.strftime("%Y-%m-01")
        st.caption("Rows per month (after filters):")
        st.table(monthly_diag)
        st.caption("Sample rows:")
        st.dataframe(df_f.head(15), use_container_width=True)

# ===================== STOP IF NO DATA =====================
if df_f.empty:
    st.warning("No data for this period with the selected filters.")
    st.stop()

# ===================== AGG + MOI =====================
g = agg_by_rep(df_f)

base_scale = MOI_SCALES[gran]
if targets_mode == "Calendar periods":
    factor = count_calendar_periods(dstart_eff, dend_eff, gran)
else:
    factor = count_observed_periods(df_f, gran, COL_DATE)

scale = _scale_for_range(base_scale, factor)
order, by_metric = _band_helpers(scale)

g["Profit Band"] = g["profit_sum"].apply(lambda v: _band_for(v, "profit", order, by_metric))
g["%Profit Band"] = g["profit_pct"].apply(lambda v: _band_for(v, "pct", order, by_metric))
g["AOV Band"] = g["aov"].apply(lambda v: _band_for(v, "aov", order, by_metric))
g["#Orders Band"] = g["orders"].apply(lambda v: _band_for(v, "orders", order, by_metric))
g["Revenue Band"] = g["revenue_sum"].apply(lambda v: _band_for(v, "revenue", order, by_metric))
g["MOI Overall"] = g.apply(lambda r: class_moi(r, scale), axis=1)

rank_map = {name: i for i, name in enumerate([t[0] for t in scale])}
g["moi_rank"] = g["MOI Overall"].map(rank_map)
g = g.sort_values(["moi_rank", "revenue_sum"], ascending=[True, False]).reset_index(drop=True)

# ===================== TITLE =====================
note = f" · targets × {factor} {gran.lower()}" + ("s" if factor != 1 else "")
st.subheader(
    f"Sales Rep — {dstart_eff} → {dend_eff}"
    + (" · Sundays excluded" if exclude_sundays else "")
    + (" · only Paid" if paid_only else "")
    + (" · negatives excluded" if exclude_neg else "")
    + note
)

# ===================== TABLE (BAND ROW + DATA ROW) =====================
rows = []
for _, r in g.iterrows():
    band_row = {
        "SALES_REP": _band_badge(r["MOI Overall"]),
        "REVENUE_SUM": _band_badge(r["Revenue Band"]),
        "PROFIT_SUM": _band_badge(r["Profit Band"]),
        "%Profit Band": _band_badge(r["%Profit Band"]),
        "ORDERS": _band_badge(r["#Orders Band"]),
        "AOV": _band_badge(r["AOV Band"]),
    }
    rows.append(band_row)
    data_row = {
        "SALES_REP": r[COL_REP],
        "REVENUE_SUM": fmt_money(r["revenue_sum"], 0),
        "PROFIT_SUM": fmt_money(r["profit_sum"], 0),
        "PROFIT_PCT": fmt_pct_unit(r["profit_pct"], 1),
        "ORDERS": int(r["orders"]),
        "AOV": fmt_money(r["aov"], 0),
    }
    rows.append(data_row)

out_interleaved = pd.DataFrame(
    rows, columns=["SALES_REP","REVENUE_SUM","PROFIT_SUM","PROFIT_PCT","ORDERS","AOV"]
)

# Render as HTML to keep badge styling
sty = out_interleaved.style
try:
    sty = sty.hide(axis="index")
except Exception:
    sty = sty.hide_index()
sty = (
    sty.set_properties(**{"white-space": "normal"})
       .set_table_styles([
            {"selector": "th", "props": [("text-align","left"), ("font-weight","700")]},
            {"selector": "td", "props": [("vertical-align","top"), ("padding","6px 8px")]}
       ])
)
html = sty.to_html(escape=False)
st.markdown(html, unsafe_allow_html=True)

# ===================== GOAL METER (Mon–Fri + única granularidad) =====================
def _period_window(base_d: date, gran: str, week_mode: str):
    ts = pd.Timestamp(base_d)
    if gran == "Day":
        return ts.normalize(), ts.normalize()
    if gran == "Week":
        if week_mode == "calendar":
            start = (ts - pd.offsets.Week(weekday=0)).normalize()
            end = (start + pd.Timedelta(days=6)).normalize()
        else:
            start = ts.normalize()
            end = (ts + pd.Timedelta(days=6)).normalize()
        return start, end
    if gran == "Month":
        p = ts.to_period("M")
        return p.start_time.normalize(), p.end_time.normalize()
    if gran == "Year":
        p = ts.to_period("Y")
        return p.start_time.normalize(), p.end_time.normalize()
    return ts.normalize(), ts.normalize()

def _revenue_in_window(df_scope: pd.DataFrame, start_ts, end_ts):
    mask = (df_scope[COL_DATE] >= start_ts) & (df_scope[COL_DATE] <= end_ts)
    return float(df_scope.loc[mask, COL_REV].sum())

def _pct(actual: float, goal: float) -> float:
    try:
        return max(0.0, min(1.0, actual / goal)) if goal and np.isfinite(goal) else 0.0
    except Exception:
        return 0.0

def meter(title: str, actual: float, goal: float):
    pct = _pct(actual, goal)
    pct_label = f"{pct*100:,.1f}%"
    bar_html = f"""
    <div style="margin:6px 0 2px 0;font-weight:700">{title}</div>
    <div style="background:#eee;border-radius:10px;overflow:hidden;height:22px;border:1px solid #ddd;">
      <div style="width:{pct*100:.4f}%;height:100%;
                  background:linear-gradient(90deg,#B7B7B7,#7F7F7F,#E06666,#A80E0E,#C00000);
                  transition:width .4s ease"></div>
    </div>
    <div style="font-size:12px;color:#444;margin-top:2px">
      {fmt_money(actual,0)} / {fmt_money(goal,0)} · <b>{pct_label}</b>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

# --- Nueva lógica de meta: tasa diaria sobre L–V y un solo medidor por granularidad ---
def workdays_between(dstart: date, dend: date, mon_fri: bool) -> int:
    s = pd.Timestamp(dstart); e = pd.Timestamp(dend)
    days = pd.date_range(s, e, freq="D")
    if mon_fri:
        days = days[days.weekday < 5]  # 0..4 = Mon–Fri
    return max(len(days), 1)

def goal_rate_per_day(goal_total: float, goal_start: date, goal_end: date, mon_fri: bool) -> float:
    n = workdays_between(goal_start, goal_end, mon_fri)
    return float(goal_total) / float(n)

def goal_for_window(rate_per_day: float, win_start: pd.Timestamp, win_end: pd.Timestamp, mon_fri: bool) -> float:
    days = pd.date_range(win_start, win_end, freq="D")
    if mon_fri:
        days = days[days.weekday < 5]
    return rate_per_day * max(len(days), 1)

# Rango total de meta (desde base_date hasta goal_end)
goal_start = pd.to_datetime(base_date).date()
goal_end_  = goal_end

# Tasa por día (Mon–Fri si está activo)
rate = goal_rate_per_day(goal_remaining, goal_start, goal_end_ , workdays_mon_fri)

# Ventana de la granularidad actual
gS, gE = _period_window(base_date, gran, week_mode)
actual_gran = _revenue_in_window(df_f, gS, gE)
goal_gran   = goal_for_window(rate, gS, gE, workdays_mon_fri)

st.markdown("### 🧪 Progress toward goal")
gran_label = f"{gran} · {pd.to_datetime(gS).date()} → {pd.to_datetime(gE).date()}"
meter(gran_label, actual_gran, goal_gran)
st.caption(
    f"Goal rate: {fmt_money(rate,0)} por día "
    + ("(Mon–Fri)" if workdays_mon_fri else "(todos los días)")
)

# ===================== NEW: DAILY COMMERCIAL ACTIVITY – COMBINED TABLE =====================
st.markdown("---")
st.markdown("## 🧩 Daily Commercial Activity – Combined Table")

# Filtro por representante (usa display name contenido en COL_REP)
rep_options = sorted(df_f[COL_REP].dropna().astype(str).unique().tolist()) if not df_f.empty else []
selected_rep = st.selectbox("Filter by Representative", options=["(All)"] + rep_options, index=0)

if selected_rep != "(All)":
    df_rep = df_f[df_f[COL_REP] == selected_rep].copy()
else:
    df_rep = df_f.copy()

# Mapeo full_name → username(s) para activity (daily_metrics.user_name) vía user_profiles
user_map_df = fetch_user_map()
if selected_rep != "(All)" and not user_map_df.empty:
    usernames_for_rep = user_map_df.loc[user_map_df["full_name"] == selected_rep, "username"].astype(str).unique().tolist()
    if not usernames_for_rep:
        # fallback por si sales_rep ya fuese username
        usernames_for_rep = [selected_rep]
else:
    usernames_for_rep = None  # sin filtro por usuario en actividad

# Actividad (Reached/Engaged/Closed) por día dentro del rango efectivo
activity_daily = fetch_daily_activity(dstart_eff, dend_eff, usernames_for_rep)

# Ventas por día (usa df_rep ya con filtros de negocio)
sales_daily = (
    df_rep.assign(d=df_rep[COL_DATE].dt.date)
          .groupby("d", dropna=False)
          .agg(
              orders=(COL_INV, pd.Series.nunique),
              revenue_sum=(COL_REV, "sum"),
              profit_sum=(COL_PROF,"sum")
          ).reset_index()
)
sales_daily["aov"] = sales_daily.apply(lambda r: _safe_div(r["revenue_sum"], r["orders"]), axis=1)
sales_daily["profit_pct"] = sales_daily.apply(lambda r: _safe_div(r["profit_sum"], r["revenue_sum"]), axis=1)

# Outer join por fecha (para no perder días sin ventas o sin actividad)
comb = pd.merge(
    sales_daily, activity_daily, on="d", how="outer", validate="1:1"
).fillna({"orders":0,"revenue_sum":0,"profit_sum":0,"aov":0,"profit_pct":0,"reached":0,"engaged":0,"closed":0})
comb = comb.sort_values("d").reset_index(drop=True)

# Tabla final (sin fila Target)
table_daily = pd.DataFrame({
    "DATE": pd.to_datetime(comb["d"]).dt.strftime("%b %d, %Y").fillna(""),
    "ORDERS CREATED": comb["orders"].astype(int),
    "Total Profit": comb["profit_sum"].apply(lambda x: fmt_money(x,0)),
    "Av. Sale": comb["aov"].apply(lambda x: fmt_money(x,0)),
    "% Profit": comb["profit_pct"].apply(lambda x: fmt_pct_unit(x,1)),
    "CLIENTS REACHED OUT TO": comb["reached"].astype(int),
    "CLIENTS ENGAGED": comb["engaged"].astype(int),
    "CLIENTS CLOSED": comb["closed"].astype(int),
})

st.dataframe(table_daily, use_container_width=True)

# Aviso de inconsistencias de input (opcional)
if (comb["engaged"] > comb["reached"]).any() or (comb["closed"] > comb["reached"]).any():
    st.warning("Some days show Engaged/Closed > Reached. Input was capped for display; please review daily_metrics entries.")

# ===================== CSV EXPORT =====================
csv_export = g.loc[:, [
    COL_REP, "MOI Overall", "revenue_sum", "profit_sum", "profit_pct",
    "orders", "aov", "Profit Band", "%Profit Band", "AOV Band", "#Orders Band", "Revenue Band"
]].rename(columns={
    COL_REP: "SALES_REP",
    "revenue_sum": "REVENUE_SUM",
    "profit_sum": "PROFIT_SUM",
    "profit_pct": "PROFIT_PCT",
    "orders": "ORDERS",
    "aov": "AOV"
})
st.download_button(
    "⬇️ Download CSV",
    csv_export.to_csv(index=False).encode("utf-8"),
    file_name=f"moi_salesrep_{pd.to_datetime(dstart_eff):%Y%m%d}_{pd.to_datetime(dend_eff):%Y%m%d}.csv",
    mime="text/csv",
)

# ===================== LEGEND =====================
st.caption("MOI Legend:")
st.markdown(
    "".join(
        f"<span style='display:inline-block;background:{col};color:{_contrast(col)};"
        f"padding:3px 8px;margin-right:6px;border-radius:6px;font-weight:600'>{name}</span>"
        for name, col in PALETA.items()
    ),
    unsafe_allow_html=True,
)

# ===================== CHARTS (kept only these) =====================
st.markdown("### 📊 Charts")

n_reps = len(g)
if n_reps == 0:
    st.info("No data to chart for this period.")
else:
    if n_reps <= 3:
        st.caption("Few reps in period: showing all (no slider).")
        topN = n_reps
    else:
        topN = st.slider("Top reps by revenue", min_value=3, max_value=min(20, n_reps), value=min(10, n_reps))
    g_top = g.head(topN)

    t_rev = alt.Tooltip("revenue_sum:Q", title="Revenue", format="$,.0f")
    t_prof = alt.Tooltip("profit_sum:Q", title="Profit", format="$,.0f")
    t_pct = alt.Tooltip("profit_pct:Q", title="Profit %", format=".1%")
    t_aov = alt.Tooltip("aov:Q", title="AOV", format="$,.0f")
    t_ord = alt.Tooltip("orders:Q", title="Orders")

    # 1) Revenue by Sales Rep
    chart_rev = (
        alt.Chart(g_top).mark_bar().encode(
            x=alt.X("revenue_sum:Q", title="Revenue", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("sales_rep:N", sort="-x", title="Sales Rep"),
            tooltip=["sales_rep", t_rev, t_prof, t_ord, t_pct, t_aov],
        ).properties(height=320, title="Revenue by Sales Rep (Top N)")
    )
    st.altair_chart(chart_rev, use_container_width=True)

    # 2) Profit by Sales Rep
    chart_profit = (
        alt.Chart(g_top).mark_bar().encode(
            x=alt.X("profit_sum:Q", title="Profit", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("sales_rep:N", sort="-x", title=None),
            tooltip=["sales_rep", t_prof, t_pct, t_rev],
        ).properties(height=260, title="Profit by Sales Rep (Top N)")
    )
    st.altair_chart(chart_profit, use_container_width=True)

    # 3) Profit % by Sales Rep
    chart_margin = (
        alt.Chart(g_top).mark_bar().encode(
            x=alt.X("profit_pct:Q", title="Profit %", axis=alt.Axis(format=".1%")),
            y=alt.Y("sales_rep:N", sort="-x", title=None),
            color=alt.Color(
                "MOI Overall:N",
                scale=alt.Scale(domain=list(PALETA.keys()), range=list(PALETA.values())),
                legend=alt.Legend(title="MOI"),
            ),
            tooltip=["sales_rep", t_pct, t_rev, t_prof, t_aov, t_ord, "MOI Overall"],
        ).properties(height=260, title="Profit % by Sales Rep (Top N)")
    )
    st.altair_chart(chart_margin, use_container_width=True)

    # 4) AOV vs Profit %
    scatter = (
        alt.Chart(g).mark_circle().encode(
            x=alt.X("aov:Q", title="AOV", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("profit_pct:Q", title="Profit %", axis=alt.Axis(format=".1%")),
            size=alt.Size("orders:Q", title="Orders", scale=alt.Scale(range=[50, 1200])),
            color=alt.Color(
                "MOI Overall:N",
                scale=alt.Scale(domain=list(PALETA.keys()), range=list(PALETA.values())),
                legend=alt.Legend(title="MOI"),
            ),
            tooltip=["sales_rep", t_aov, t_pct, t_ord, t_rev, t_prof, "MOI Overall"],
        ).properties(height=340, title="AOV vs Profit % (size = orders)")
    )
    st.altair_chart(scatter, use_container_width=True)
