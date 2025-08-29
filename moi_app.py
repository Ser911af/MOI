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
    """Reads first from st.secrets (Streamlit Cloud), then from .env.
    Falls back to provided default.
    """
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_ANON_KEY")
SUPABASE_TABLE = _get_secret("SUPABASE_TABLE", "ventas_frutto")
LOGO_PATH = _get_secret(
    "LOGO_PATH", "Logo/WhatsApp Image 2025-08-26 at 1.50.59 PM (1).jpeg"
)

# ===================== PASSWORD (single user) =====================
# Define the password in .env or in .streamlit/secrets.toml as MOI_PASSWORD
APP_PASSWORD = _get_secret("MOI_PASSWORD", None)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 MOI – Acceso restringido")
    if not APP_PASSWORD:
        st.warning(
            "No se encontró MOI_PASSWORD en secrets/.env. Define una contraseña segura."
        )
    pwd = st.text_input("Introduce la contraseña:", type="password")
    if st.button("Entrar"):
        if APP_PASSWORD and pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Acceso concedido ✅")
            st.rerun()
        else:
            st.error("❌ Contraseña incorrecta")
    st.stop()  # detiene la app si no está autenticado

# ===================== APP NORMAL (solo visible tras login) =====================

COL_DATE = "reqs_date"
COL_REP = "sales_rep"
COL_INV = "invoice_num"
COL_REV = "total_revenue"
COL_PROF = "total_profit_usd"
COL_STATUS = "invoice_payment_status"

NEEDED = [COL_DATE, COL_REP, COL_INV, COL_REV, COL_PROF, COL_STATUS]

# ===== Formatting

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


# ===== MOI Palette

PALETA = {
    "Remarkable": "#C00000",
    "Excellent": "#A80E0E",
    "Great": "#E06666",
    "Good": "#4D4D4D",
    "Average": "#7F7F7F",
    "Poor": "#B7B7B7",
}


# ===== Color helpers for Styler cells (bands and MOI Overall)

def _contrast(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 140 else "#000000"


def style_moi_bands(df: pd.DataFrame):
    band_cols = [
        "Profit Band",
        "%Profit Band",
        "AOV Band",
        "#Orders Band",
        "Revenue Band",
        "MOI Overall",
    ]
    sty = df.style
    for c in band_cols:
        if c in df.columns:
            sty = sty.map(
                lambda v: (
                    ""
                    if pd.isna(v)
                    else f"background-color:{PALETA.get(v, '#FFFFFF')};"
                    f"color:{_contrast(PALETA.get(v, '#FFFFFF'))};"
                    "font-weight:600;"
                ),
                subset=pd.IndexSlice[:, [c]],
            )
    return sty


# ===== Base MOI scales

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


# ===== MOI helpers

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
    return max(int((days.weekday != 6).sum()), 1)  # exclude Sundays when counting days


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


# ===================== DATA HELPERS =====================

def effective_range(d: date, granularity: str, week_mode: str = "window"):
    ts = pd.Timestamp(d)
    if granularity == "Day":
        return ts.date(), ts.date()
    if granularity == "Week":
        if week_mode == "window":
            dstart = ts.date()
            dend = (ts + pd.Timedelta(days=6)).date()
        else:
            dstart = (ts - pd.offsets.Week(weekday=0)).date()  # Monday
            dend = (pd.Timestamp(dstart) + pd.Timedelta(days=6)).date()
        return dstart, dend
    if granularity == "Month":
        p = ts.to_period("M")
        return p.start_time.date(), p.end_time.date()
    if granularity == "Year":
        p = ts.to_period("Y")
        return p.start_time.date(), p.end_time.date()
    return ts.date(), ts.date()


# ====== NEW: fetch_server_filtered_v2 con conteo exacto y manejo de tz ======
@st.cache_data(show_spinner=False, ttl=60)
def fetch_server_filtered_v2(dstart: date, dend: date) -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.session_state["_last_fetch_rows"] = 0
        st.session_state["_last_fetch_pages"] = 0
        st.session_state["_last_fetch_err"] = "Missing Supabase secrets"
        st.session_state["_server_count"] = None
        return pd.DataFrame(columns=NEEDED)

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    start_s = pd.Timestamp(dstart).date().isoformat()
    end_exclusive = (pd.Timestamp(dend) + pd.Timedelta(days=1)).date().isoformat()

    frames: list[pd.DataFrame] = []
    pages = 0
    start = 0
    # ⚠️ Supabase/PostgREST limita el tamaño de rango por request a ~1000 filas.
    # Si pedimos 2000, devuelve 1000 y al sumar 2000 saltamos 1000 filas -> pérdida de datos.
    PAGE = 1000
    select_list = ",".join(NEEDED)

    # Primera llamada: obtener count exacto y la primera página
    base = (
        client.table(SUPABASE_TABLE)
        .select(select_list, count="exact")
        .gte(COL_DATE, start_s)
        .lt(COL_DATE, end_exclusive)
        .order(COL_DATE, desc=False)
    )

    first_last = start + PAGE - 1
    resp = base.range(start, first_last).execute()
    try:
        total_count = resp.count
    except Exception:
        total_count = None

    batch = pd.DataFrame(resp.data or [])
    if not batch.empty:
        frames.append(batch)
        pages += 1
        start += len(batch)  # avanzamos exactamente lo que recibimos

    # Paginamos hasta consumir total_count (si lo sabemos) o hasta que vengan páginas vacías/pequeñas
    while True:
        if total_count is not None and start >= total_count:
            break
        last = start + PAGE - 1
        resp = base.range(start, last).execute()
        batch = pd.DataFrame(resp.data or [])
        if batch.empty:
            break
        frames.append(batch)
        pages += 1
        got = len(batch)
        start += got
        if got < PAGE:
            # última página
            break

    if not frames:
        st.session_state["_last_fetch_rows"] = 0
        st.session_state["_last_fetch_pages"] = 0
        st.session_state["_last_fetch_err"] = None
        st.session_state["_server_count"] = total_count
        return pd.DataFrame(columns=NEEDED)

    df = pd.concat(frames, ignore_index=True)

    # Fecha: evitar desplazamientos por tz
    if COL_DATE in df.columns:
        s = pd.to_datetime(df[COL_DATE], errors="coerce", utc=False)
        try:
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert(None)
        except Exception:
            pass
        df[COL_DATE] = s

    df[COL_REV] = pd.to_numeric(df[COL_REV], errors="coerce")
    df[COL_PROF] = pd.to_numeric(df[COL_PROF], errors="coerce")
    if COL_REP in df.columns:
        df[COL_REP] = df[COL_REP].astype("string").str.strip().fillna("(No rep)")

    for c in NEEDED:
        if c not in df.columns:
            df[c] = np.nan

    df = df[NEEDED].dropna(subset=[COL_DATE]).reset_index(drop=True)

    st.session_state["_last_fetch_rows"] = len(df)
    st.session_state["_last_fetch_pages"] = pages
    st.session_state["_last_fetch_err"] = None
    st.session_state["_server_count"] = total_count

    return df


# ====== NEW fetch_server_filtered (inclusive end via lt(dend+1), TTL, logging) ======
@st.cache_data(show_spinner=False, ttl=60)  # pequeño TTL para evitar cache eterno
def fetch_server_filtered(dstart: date, dend: date) -> pd.DataFrame:
    # ---- Guard clause: si faltan secrets, devolvemos DF vacío con columnas esperadas
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.session_state["_last_fetch_rows"] = 0
        st.session_state["_last_fetch_pages"] = 0
        st.session_state["_last_fetch_err"] = "Missing Supabase secrets"
        return pd.DataFrame(columns=NEEDED)

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ---- Ventana de fechas inclusiva y robusta (funciona con DATE o TIMESTAMP):
    # >= dstart  y  < (dend + 1 día)  => incluye TODO el último día aunque haya horas
    start_s = pd.Timestamp(dstart).date().isoformat()
    end_exclusive = (pd.Timestamp(dend) + pd.Timedelta(days=1)).date().isoformat()

    frames = []
    pages = 0
    start = 0
    PAGE = 2000
    select_list = ",".join(NEEDED)

    while True:
        last = start + PAGE - 1
        resp = (
            client.table(SUPABASE_TABLE)
            .select(select_list)
            .gte(COL_DATE, start_s)  # >= dstart
            .lt(COL_DATE, end_exclusive)  #  < dend+1 día (fin inclusivo)
            .order(COL_DATE, desc=False)
            .range(start, last)  # paginación 0-based, inclusiva
            .execute()
        )
        batch = pd.DataFrame(resp.data or [])
        if batch.empty:
            break

        frames.append(batch)
        pages += 1

        # si la página viene incompleta, ya no hay más
        if len(batch) < PAGE:
            break

        # avanza a la siguiente página
        start += PAGE

    # ---- Si no hubo datos, retorna vacío con columnas esperadas (evita KeyError)
    if not frames:
        st.session_state["_last_fetch_rows"] = 0
        st.session_state["_last_fetch_pages"] = 0
        st.session_state["_last_fetch_err"] = None
        return pd.DataFrame(columns=NEEDED)

    # ---- Concatena y normaliza tipos
    df = pd.concat(frames, ignore_index=True)

    if COL_DATE in df.columns:
        # parsea timestamps y quita tz para que .dt funcione de forma local
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", utc=True).dt.tz_convert(None)
    df[COL_REV] = pd.to_numeric(df[COL_REV], errors="coerce")
    df[COL_PROF] = pd.to_numeric(df[COL_PROF], errors="coerce")
    if COL_REP in df.columns:
        df[COL_REP] = df[COL_REP].astype("string").str.strip().fillna("(No rep)")

    for c in NEEDED:
        if c not in df.columns:
            df[c] = np.nan

    df = df[NEEDED].dropna(subset=[COL_DATE]).reset_index(drop=True)

    # ---- Logging para auditar en la UI
    st.session_state["_last_fetch_rows"] = len(df)
    st.session_state["_last_fetch_pages"] = pages
    st.session_state["_last_fetch_err"] = None

    return df


def apply_filters(df: pd.DataFrame, exclude_sundays: bool, paid_only: bool, exclude_neg: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=NEEDED)
    # Ensure required columns exist and correct dtype
    for c in NEEDED:
        if c not in df.columns:
            df[c] = pd.Series(index=df.index, dtype="float64")
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


def agg_by_rep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "sales_rep",
                "revenue_sum",
                "profit_sum",
                "orders",
                "profit_pct",
                "aov",
            ]
        )
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
    g["aov"] = np.where(g["orders"] > 0, g["revenue_sum"] / g["orders"], 0.0)
    return g.sort_values(["revenue_sum", "orders"], ascending=[False, False])


# ===================== UI – Header =====================

# Header with logo
c1, c2 = st.columns([1, 6])
with c1:
    logo_file = Path(LOGO_PATH)
    if logo_file.exists():
        try:
            from PIL import Image  # optional dependency

            st.image(Image.open(logo_file), use_container_width=True)
        except Exception:
            st.empty()
with c2:
    st.title("MOI – Sales Rep View (server-side date filter)")

# ===================== Sidebar =====================

with st.sidebar:
    st.markdown("### ⚙️ Debug")
    debug_ignore_filters = st.checkbox(
        "Debug: ignore all business filters",
        value=False,
        help="Aplica ningún filtro (domingos, paid, negativos) para comparar con SQL.",
    )

    gran = st.radio("Granularity", ["Day", "Week", "Month", "Year"], horizontal=True)
    week_mode = st.radio(
        "Week mode",
        ["window", "calendar"],
        index=0,
        help="window: 7 days starting from the selected date · calendar: Monday to Sunday",
    )
    base_date = st.date_input("Base date", value=date.today())

    st.markdown("**Targets scale based on…**")
    targets_mode = st.radio(
        "", ["Calendar periods", "Observed periods"],
        help=(
            "Calendar: targets × number of calendar periods in range · Observed: × periods with data"
        ),
    )

    st.markdown("**Business filters (apply to period):**")
    exclude_sundays = st.checkbox("Exclude Sundays", value=True)
    paid_only = st.checkbox("Only invoices with status 'Paid'", value=False)
    exclude_neg = st.checkbox(
        "Exclude negatives (revenue ≤ 0 or profit < 0)", value=True
    )

    if st.button("Refresh"):
        st.rerun()

# Compute effective range

dstart_eff, dend_eff = effective_range(base_date, gran, week_mode)
st.caption(f"**Effective range:** {dstart_eff} → {dend_eff}")

# ===================== Health Check =====================

with st.expander("🩺 Health check", expanded=False):
    st.write(
        {
            "has_SUPABASE_URL": bool(SUPABASE_URL),
            "has_SUPABASE_ANON_KEY": bool(SUPABASE_KEY),
            "table": SUPABASE_TABLE,
            "logo_exists": Path(LOGO_PATH).exists(),
        }
    )

# ===================== Fetch & filters =====================

with st.spinner("Querying Supabase…"):
    df_raw = fetch_server_filtered_v2(dstart_eff, dend_eff)
    df_f = apply_filters(df_raw, exclude_sundays, paid_only, exclude_neg)

# ===================== Diagnostics =====================

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
            "server_count": st.session_state.get("_server_count"),
            "fetched_rows": st.session_state.get("_last_fetch_rows"),
            "pages": st.session_state.get("_last_fetch_pages"),
        }
    )
    if st.session_state.get("_server_count") is not None:
        delta = (st.session_state.get("_server_count") or 0) - (
            st.session_state.get("_last_fetch_rows") or 0
        )
        if delta != 0:
            st.warning(
                f"⚠️ Server count says {st.session_state['_server_count']} rows for the range, but we fetched {st.session_state['_last_fetch_rows']}. Δ = {delta}. Revisa filtros, paginación o tipos de fecha."
            )

    if not df_f.empty:
        monthly = (
            df_f.assign(month=df_f[COL_DATE].dt.to_period("M").dt.start_time)
            .groupby("month")
            .size()
            .reset_index(name="rows")
        )
        monthly["month"] = monthly["month"].dt.strftime("%Y-%m-01")
        st.caption("Rows per month (after filters):")
        st.table(monthly)
        st.caption("Sample rows:")
        st.dataframe(df_f.head(15), use_container_width=True)

# ===================== Ranking + MOI =====================

if df_f.empty:
    st.warning(
        "No data for this period with the selected filters. Try widening the range or unchecking filters."
    )
    st.stop()


g = agg_by_rep(df_f)

# Targets factor & bands
base_scale = MOI_SCALES[gran]
if targets_mode == "Calendar periods":
    factor = count_calendar_periods(dstart_eff, dend_eff, gran)
else:
    factor = count_observed_periods(df_f, gran, COL_DATE)

scale = _scale_for_range(base_scale, factor)
order, by_metric = _band_helpers(scale)


g["Profit Band"] = g["profit_sum"].apply(
    lambda v: _band_for(v, "profit", order, by_metric)
)
g["%Profit Band"] = g["profit_pct"].apply(lambda v: _band_for(v, "pct", order, by_metric))
g["AOV Band"] = g["aov"].apply(lambda v: _band_for(v, "aov", order, by_metric))
g["#Orders Band"] = g["orders"].apply(lambda v: _band_for(v, "orders", order, by_metric))
g["Revenue Band"] = g["revenue_sum"].apply(
    lambda v: _band_for(v, "revenue", order, by_metric)
)
g["MOI Overall"] = g.apply(lambda r: class_moi(r, scale), axis=1)

rank_map = {name: i for i, name in enumerate([t[0] for t in scale])}
g["moi_rank"] = g["MOI Overall"].map(rank_map)
g = g.sort_values(["moi_rank", "revenue_sum"], ascending=[True, False]).reset_index(
    drop=True
)

# ---- formatted output
out = pd.DataFrame(
    {
        "SALES_REP": g[COL_REP],
        "MOI Overall": g["MOI Overall"],
        "REVENUE_SUM": g["revenue_sum"].map(lambda x: fmt_money(x, 0)),
        "PROFIT_SUM": g["profit_sum"].map(lambda x: fmt_money(x, 0)),
        "PROFIT_PCT": g["profit_pct"].map(lambda x: fmt_pct_unit(x, 1)),
        "ORDERS": g["orders"].astype(int),
        "AOV": g["aov"].map(lambda x: fmt_money(x, 0)),
        "Profit Band": g["Profit Band"],
        "%Profit Band": g["%Profit Band"],
        "AOV Band": g["AOV Band"],
        "#Orders Band": g["#Orders Band"],
        "Revenue Band": g["Revenue Band"],
    }
)

note = f" · targets × {factor} {gran.lower()}" + ("s" if factor != 1 else "")
st.subheader(
    f"Sales Rep — {dstart_eff} → {dend_eff}"
    + (" · Sundays excluded" if exclude_sundays else "")
    + (" · only Paid" if paid_only else "")
    + (" · negatives excluded" if exclude_neg else "")
    + note
)

# Render styled table
styled_out = style_moi_bands(out)
st.dataframe(styled_out, use_container_width=True)

# CSV download (unstyled)
st.download_button(
    "⬇️ Download CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name=f"moi_salesrep_{pd.to_datetime(dstart_eff):%Y%m%d}_{pd.to_datetime(dend_eff):%Y%m%d}.csv",
    mime="text/csv",
)

# Color legend
st.caption("MOI Legend:")
st.markdown(
    "".join(
        f"<span style='display:inline-block;background:{col};color:{_contrast(col)};"
        f"padding:3px 8px;margin-right:6px;border-radius:6px;font-weight:600'>{name}</span>"
        for name, col in PALETA.items()
    ),
    unsafe_allow_html=True,
)

# ===================== Charts =====================

st.markdown("### 📊 Charts")

n_reps = len(g)
if n_reps == 0:
    st.info("No data to chart for this period.")
else:
    if n_reps <= 3:
        st.caption("Few reps in period: showing all (no slider).")
        topN = n_reps
    else:
        topN = st.slider(
            "Top reps by revenue", min_value=3, max_value=min(20, n_reps), value=min(10, n_reps)
        )

    g_top = g.head(topN)

    t_rev = alt.Tooltip("revenue_sum:Q", title="Revenue", format="$,.0f")
    t_prof = alt.Tooltip("profit_sum:Q", title="Profit", format="$,.0f")
    t_pct = alt.Tooltip("profit_pct:Q", title="Profit %", format=".1%")
    t_aov = alt.Tooltip("aov:Q", title="AOV", format="$,.0f")
    t_ord = alt.Tooltip("orders:Q", title="Orders")

    if not g_top.empty:
        chart_rev = (
            alt.Chart(g_top)
            .mark_bar()
            .encode(
                x=alt.X("revenue_sum:Q", title="Revenue", axis=alt.Axis(format="$,.0f")),
                y=alt.Y("sales_rep:N", sort="-x", title="Sales Rep"),
                tooltip=["sales_rep", t_rev, t_prof, t_ord, t_pct, t_aov],
            )
            .properties(height=320, title="Revenue by Sales Rep (Top N)")
        )
        st.altair_chart(chart_rev, use_container_width=True)

        chart_profit = (
            alt.Chart(g_top)
            .mark_bar()
            .encode(
                x=alt.X("profit_sum:Q", title="Profit", axis=alt.Axis(format="$,.0f")),
                y=alt.Y("sales_rep:N", sort="-x", title=None),
                tooltip=["sales_rep", t_prof, t_pct, t_rev],
            )
            .properties(height=260, title="Profit by Sales Rep (Top N)")
        )
        st.altair_chart(chart_profit, use_container_width=True)

        chart_margin = (
            alt.Chart(g_top)
            .mark_bar()
            .encode(
                x=alt.X("profit_pct:Q", title="Profit %", axis=alt.Axis(format=".1%")),
                y=alt.Y("sales_rep:N", sort="-x", title=None),
                color=alt.Color(
                    "MOI Overall:N",
                    scale=alt.Scale(
                        domain=list(PALETA.keys()), range=list(PALETA.values())
                    ),
                    legend=alt.Legend(title="MOI"),
                ),
                tooltip=[
                    "sales_rep",
                    t_pct,
                    t_rev,
                    t_prof,
                    t_aov,
                    t_ord,
                    "MOI Overall",
                ],
            )
            .properties(height=260, title="Profit % by Sales Rep (Top N)")
        )
        st.altair_chart(chart_margin, use_container_width=True)
    else:
        st.info("Not enough reps for comparison charts.")


daily = (
    df_f.assign(day=df_f[COL_DATE].dt.floor("D"))
    .groupby("day", as_index=False)
    .agg(
        revenue=(COL_REV, "sum"),
        profit=(COL_PROF, "sum"),
        orders=(COL_INV, pd.Series.nunique),
    )
)
if not daily.empty:
    line_rev = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("revenue:Q", title="Daily revenue", axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("day:T", title="Date"),
                alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f"),
                alt.Tooltip("profit:Q", title="Profit", format="$,.0f"),
                alt.Tooltip("orders:Q", title="Orders"),
            ],
        )
        .properties(height=300, title="Daily revenue")
    )
    st.altair_chart(line_rev, use_container_width=True)

by_day_rep = (
    df_f.assign(day=df_f[COL_DATE].dt.floor("D"))
    .groupby(["day", "sales_rep"], as_index=False)
    .agg(revenue=(COL_REV, "sum"))
)
if not by_day_rep.empty:
    stack_day = (
        alt.Chart(by_day_rep)
        .mark_bar()
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("revenue:Q", title="Revenue", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("sales_rep:N", title="Sales Rep"),
            tooltip=[
                alt.Tooltip("day:T", title="Date"),
                "sales_rep",
                alt.Tooltip("revenue:Q", title="Revenue", format="$,.0f"),
            ],
        )
        .properties(height=320, title="Revenue by day and rep (stacked)")
    )
    st.altair_chart(stack_day, use_container_width=True)

if not g.empty:
    scatter = (
        alt.Chart(g)
        .mark_circle()
        .encode(
            x=alt.X("aov:Q", title="AOV", axis=alt.Axis(format="$,.0f")),
            y=alt.Y("profit_pct:Q", title="Profit %", axis=alt.Axis(format=".1%")),
            size=alt.Size("orders:Q", title="Orders", scale=alt.Scale(range=[50, 1200])),
            color=alt.Color(
                "MOI Overall:N",
                scale=alt.Scale(domain=list(PALETA.keys()), range=list(PALETA.values())),
                legend=alt.Legend(title="MOI"),
            ),
            tooltip=["sales_rep", t_aov, t_pct, t_ord, t_rev, t_prof, "MOI Overall"],
        )
        .properties(height=340, title="AOV vs Profit % (size = orders)")
    )
    st.altair_chart(scatter, use_container_width=True)
