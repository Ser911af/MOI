import os
from datetime import date
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from pathlib import Path
from PIL import Image

# ===================== CONFIG =====================

st.set_page_config(page_title="Frutto – MOI Dashboard", layout="wide")
load_dotenv()

# Helpers para secrets/env
def _get_secret(name: str, default=None):
    try:
        if hasattr(st, "secrets") and name in st.secrets:  # Streamlit Cloud
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

SUPABASE_URL   = _get_secret("SUPABASE_URL")
SUPABASE_KEY   = _get_secret("SUPABASE_ANON_KEY")
SUPABASE_TABLE = _get_secret("SUPABASE_TABLE", "ventas_frutto")
LOGO_PATH      = _get_secret("LOGO_PATH", "Logo/WhatsApp Image 2025-08-26 at 1.50.59 PM (1).jpeg")

COL_DATE   = "reqs_date"
COL_REP    = "sales_rep"
COL_INV    = "invoice_num"
COL_REV    = "total_revenue"
COL_PROF   = "total_profit_usd"
COL_STATUS = "invoice_payment_status"

NEEDED = [COL_DATE, COL_REP, COL_INV, COL_REV, COL_PROF, COL_STATUS]

# ===== Formatos

def fmt_money(v, decimals: int = 0):  # $ con separadores
    try:
        return f"${v:,.{decimals}f}"
    except Exception:
        return v


def fmt_pct_unit(v, decimals: int = 1):  # 0.134 -> "13.4%"
    try:
        return f"{(100*v):.{decimals}f}%"
    except Exception:
        return v

# ===== Paleta MOI

PALETA = {
    "Remarkable": "#C00000", "Excellent": "#A80E0E", "Great": "#E06666",
    "Good": "#4D4D4D", "Average": "#7F7F7F", "Poor": "#B7B7B7",
}

# ===== Helpers de color para celdas (bandas y MOI Overall)

def _contrast(hex_color: str) -> str:
    """Texto blanco/negro según luminosidad para que siempre se lea bien."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # luminancia simple (suficiente para UI)
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return "#FFFFFF" if lum < 140 else "#000000"


def style_moi_bands(df: pd.DataFrame):
    band_cols = ["Profit Band", "%Profit Band", "AOV Band", "#Orders Band", "Revenue Band", "MOI Overall"]
    sty = df.style
    for c in band_cols:
        if c in df.columns:
            sty = sty.map(
                lambda v: (
                    "" if pd.isna(v)
                    else f"background-color:{PALETA.get(v, '#FFFFFF')};"
                         f"color:{_contrast(PALETA.get(v, '#FFFFFF'))};"
                         "font-weight:600;"
                ),
                subset=pd.IndexSlice[:, [c]],
            )
    return sty

# ===== Escalas MOI base (por período)

MOI_SCALES = {
    "Year": [
        ("Remarkable", 2000000, 0.20, 11000, 900, 10000000),
        ("Excellent",  1520000, 0.19, 10000, 780,  8000000),
        ("Great",      1080000, 0.18,  9000, 660,  6000000),
        ("Good",        850000, 0.17,  8000, 600,  5000000),
        ("Average",     640000, 0.16,  7000, 540,  4000000),
        ("Poor",        450000, 0.15,  6000, 480,  3000000),
    ],
    "Month": [
        ("Remarkable", 166667, 0.20, 11000, 75, 833333),
        ("Excellent",  126667, 0.19, 10000, 65, 666667),
        ("Great",       90000, 0.18,  9000, 55, 500000),
        ("Good",        70833, 0.17,  8000, 50, 416667),
        ("Average",     53333, 0.16,  7000, 45, 333333),
        ("Poor",        37500, 0.15,  6000, 40, 250000),
    ],
    "Week": [
        ("Remarkable", 38462, 0.20, 11000, 34, 192308),
        ("Excellent",  29231, 0.19, 10000, 28, 153846),
        ("Great",      20769, 0.18,  9000, 23, 115385),
        ("Good",       16346, 0.17,  8000, 20,  96154),
        ("Average",    12308, 0.16,  7000, 17,  76923),
        ("Poor",         8654, 0.15,  6000, 14,  57692),
    ],
    "Day": [
        ("Remarkable",  5479, 0.20, 11000, 7, 27397),
        ("Excellent",   4164, 0.19, 10000, 6, 21918),
        ("Great",       2959, 0.18,  9000, 5, 16438),
        ("Good",        2329, 0.17,  8000, 4, 13699),
        ("Average",     1753, 0.16,  7000, 4, 10959),
        ("Poor",        1233, 0.15,  6000, 3,  8219),
    ],
}

# ===== Helpers MOI

def _scale_for_range(base_scale, factor: float):
    # Escala revenue/profit/órdenes por factor; % y AOV no cambian
    return [(name, p*factor, pct, aov, ords*factor, rev*factor)
            for (name, p, pct, aov, ords, rev) in base_scale]


def _band_helpers(scale):
    order = [t[0] for t in scale]
    by_metric = {
        "profit":  {n: p   for (n, p, _pct, _aov, _ord, _rev) in scale},
        "pct":     {n: pct for (n, _p, pct, _aov, _ord, _rev) in scale},
        "aov":     {n: aov for (n, _p, _pct, aov, _ord, _rev) in scale},
        "orders":  {n: o   for (n, _p, _pct, _aov, o, _rev) in scale},
        "revenue": {n: r   for (n, _p, _pct, _aov, _ord, r) in scale},
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
        checks += row["profit_sum"]  >= min_profit
        checks += row["profit_pct"]  >= min_pct
        checks += row["aov"]         >= min_aov
        checks += row["orders"]      >= min_orders
        if reinforce_revenue:
            checks += row["revenue_sum"] >= min_rev
        if checks >= majority_required + 1:
            return name
        else:
            if checks >= majority_required:
                return name
    return scale[-1][0]


def count_calendar_periods(dstart, dend, gran):
    s = pd.Timestamp(dstart); e = pd.Timestamp(dend)
    if gran == "Year":
        return max(len(pd.period_range(s, e, freq="Y")), 1)
    if gran == "Month":
        return max(len(pd.period_range(s, e, freq="M")), 1)
    if gran == "Week":
        return max(len(pd.period_range(s, e, freq="W-MON")), 1)
    # Day = días L–S
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

# ===================== HELPERS DATA =====================

def rango_efectivo(fecha: date, granularidad: str, modo_semana: str = "ventana"):
    ts = pd.Timestamp(fecha)
    if granularidad == "Day":
        return ts.date(), ts.date()
    if granularidad == "Week":
        # requested: la fecha seleccionada es el PRIMER día
        if modo_semana == "ventana":
            dstart = ts.date()
            dend   = (ts + pd.Timedelta(days=6)).date()
        else:
            dstart = (ts - pd.offsets.Week(weekday=0)).date()  # lunes de esa semana
            dend   = (pd.Timestamp(dstart) + pd.Timedelta(days=6)).date()
        return dstart, dend
    if granularidad == "Month":
        p = ts.to_period("M")
        return p.start_time.date(), p.end_time.date()
    if granularidad == "Year":
        p = ts.to_period("Y")
        return p.start_time.date(), p.end_time.date()
    return ts.date(), ts.date()


@st.cache_data(show_spinner=False)
def fetch_server_filtered(dstart: date, dend: date) -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame()

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    frames, start = [], 0
    PAGE = 2000
    select_list = ",".join(NEEDED)

    while True:
        resp = (
            client.table(SUPABASE_TABLE)
                  .select(select_list)
                  .gte(COL_DATE, dstart.isoformat())
                  .lte(COL_DATE, dend.isoformat())
                  .order(COL_DATE, desc=False)
                  .range(start, start + PAGE - 1)
                  .execute()
        )
        batch = pd.DataFrame(resp.data or [])
        if batch.empty:
            break
        frames.append(batch)
        if len(batch) < PAGE:
            break
        start += PAGE

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df[COL_REV]  = pd.to_numeric(df[COL_REV], errors="coerce")
    df[COL_PROF] = pd.to_numeric(df[COL_PROF], errors="coerce")

    rep = df[COL_REP].astype("string").str.strip().fillna("(Sin rep)")
    df[COL_REP] = rep

    for c in NEEDED:
        if c not in df.columns:
            df[c] = np.nan

    return df[NEEDED].dropna(subset=[COL_DATE]).reset_index(drop=True)


def aplicar_filtros(df: pd.DataFrame, excluir_domingos: bool, paid_only: bool, excluir_neg: bool) -> pd.DataFrame:
    x = df.copy()
    if excluir_domingos:
        x = x[x[COL_DATE].dt.weekday != 6]
    if paid_only:
        x = x[x[COL_STATUS] == "Paid"]
    if excluir_neg:
        x = x[(x[COL_REV] > 0) & (x[COL_PROF] >= 0)]
    return x


def agg_by_rep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sales_rep", "revenue_sum", "profit_sum", "orders", "profit_pct", "aov"])
    g = (
        df.groupby(COL_REP, dropna=False)
          .agg(revenue_sum=(COL_REV, "sum"),
               profit_sum=(COL_PROF, "sum"),
               orders=(COL_INV, pd.Series.nunique))
          .reset_index()
    )
    g["profit_pct"] = np.where(g["revenue_sum"] > 0, g["profit_sum"]/g["revenue_sum"], 0.0)
    g["aov"]        = np.where(g["orders"] > 0, g["revenue_sum"]/g["orders"], 0.0)
    return g.sort_values(["revenue_sum", "orders"], ascending=[False, False])

# ===================== UI – Sidebar =====================

# Logo + título
logo_file = Path(LOGO_PATH)
cols = st.columns([1,6])
with cols[0]:
    if logo_file.exists():
        try:
            st.image(Image.open(logo_file), use_container_width=True)
        except Exception:
            st.empty()
with cols[1]:
    st.title("MOI simple — por Sales Rep (base: reqs_date, filtro en servidor)")

with st.sidebar:
    gran = st.radio("Granularidad", ["Day", "Week", "Month", "Year"], horizontal=True)
    modo_semana = st.radio(
        "Semana basada en…", ["ventana", "calendario"], index=0,
        help="‘ventana’ = 7 días desde la fecha elegida; ‘calendario’ = lunes a domingo",
    )
    fecha_base = st.date_input("Elige fecha base", value=date.today())

    st.markdown("**Metas basadas en…**")
    modo_metas = st.radio(
        "",
        ["Calendario", "Períodos observados"],
        help=(
            "Calendario: metas × cantidad de periodos del calendario en el rango. "
            "Observados: × periodos con datos."
        ),
    )

    st.markdown("**Filtros de negocio (aplican al período):**")
    excluir_domingos = st.checkbox("Excluir domingos", value=True)
    paid_only        = st.checkbox("Solo facturas 'Paid'", value=False)
    excluir_neg      = st.checkbox("Excluir negativos (revenue ≤ 0 o profit < 0)", value=True)

    if st.button("Refrescar"):
        st.rerun()


dstart_eff, dend_eff = rango_efectivo(fecha_base, gran, modo_semana)
st.caption(f"**Rango efectivo:** {dstart_eff} → {dend_eff}")

# ===================== Fetch & filtros =====================

with st.spinner("Consultando Supabase…"):
    df_raw = fetch_server_filtered(dstart_eff, dend_eff)

    df_f = aplicar_filtros(df_raw, excluir_domingos, paid_only, excluir_neg)

# ===================== Diagnóstico =====================

with st.expander("🔎 Diagnóstico", expanded=False):
    st.write({
        "granularidad": gran,
        "modo_semana": modo_semana,
        "metas": modo_metas,
        "desde": str(dstart_eff),
        "hasta": str(dend_eff),
        "filas_df_raw": int(len(df_raw)),
        "filas_post_filtros": int(len(df_f)),
        "reps_unicos": int(df_f[COL_REP].nunique()) if not df_f.empty else 0,
        "min_req": df_f[COL_DATE].min().strftime("%Y-%m-%d") if not df_f.empty else None,
        "max_req": df_f[COL_DATE].max().strftime("%Y-%m-%d") if not df_f.empty else None,
    })
    if not df_f.empty:
        monthly = (
            df_f.assign(month=df_f[COL_DATE].dt.to_period("M").dt.start_time)
                .groupby("month").size().reset_index(name="rows")
        )
        monthly["month"] = monthly["month"].dt.strftime("%Y-%m-01")
        st.caption("Filas por mes (post-filtros):")
        st.table(monthly)
        st.caption("Muestra de filas:")
        st.dataframe(df_f.head(15), use_container_width=True)

# ===================== Ranking + MOI =====================

if df_f.empty:
    st.warning("No hay datos para ese período con los filtros aplicados.")
    st.stop()

g = agg_by_rep(df_f)

# ---- calcular factor metas y bandas

base_scale = MOI_SCALES[gran]
if modo_metas == "Calendario":
    factor = count_calendar_periods(dstart_eff, dend_eff, gran)
else:
    factor = count_observed_periods(df_f, gran, COL_DATE)

scale = _scale_for_range(base_scale, factor)
order, by_metric = _band_helpers(scale)

g["Profit Band"]   = g["profit_sum"].apply(lambda v: _band_for(v, "profit",  order, by_metric))
g["%Profit Band"]  = g["profit_pct"].apply(lambda v: _band_for(v, "pct",     order, by_metric))
g["AOV Band"]      = g["aov"].apply(lambda v: _band_for(v, "aov",           order, by_metric))
g["#Orders Band"]  = g["orders"].apply(lambda v: _band_for(v, "orders",      order, by_metric))
g["Revenue Band"]  = g["revenue_sum"].apply(lambda v: _band_for(v, "revenue", order, by_metric))
g["MOI Overall"]   = g.apply(lambda r: class_moi(r, scale), axis=1)

rank_map = {name: i for i, name in enumerate([t[0] for t in scale])}
g["moi_rank"] = g["MOI Overall"].map(rank_map)
g = g.sort_values(["moi_rank", "revenue_sum"], ascending=[True, False]).reset_index(drop=True)

# ---- salida formateada

out = pd.DataFrame({
    "SALES_REP": g[COL_REP],
    "MOI Overall": g["MOI Overall"],
    "REVENUE_SUM": g["revenue_sum"].map(lambda x: fmt_money(x, 0)),
    "PROFIT_SUM": g["profit_sum"].map(lambda x: fmt_money(x, 0)),
    "PROFIT_PCT": g["profit_pct"].map(lambda x: fmt_pct_unit(x, 1)),
    "ORDERS": g["orders"].astype(int),
    "AOV": g["aov"].map(lambda x: fmt_money(x, 0)),
    "Profit Band":  g["Profit Band"],
    "%Profit Band": g["%Profit Band"],
    "AOV Band":     g["AOV Band"],
    "#Orders Band": g["#Orders Band"],
    "Revenue Band": g["Revenue Band"],
})

nota = (f" · metas × {factor} {gran.lower()}" + ("s" if factor != 1 else ""))
st.subheader(
    f"Sales Rep — {dstart_eff} → {dend_eff}"
    + (" · domingos excluidos" if excluir_domingos else "")
    + (" · solo Paid" if paid_only else "")
    + (" · negativos excluidos" if excluir_neg else "")
    + nota
)

# Render con estilos de bandas
styled_out = style_moi_bands(out)
st.dataframe(styled_out, use_container_width=True)

# Descarga (desde 'out' sin estilos)
st.download_button(
    "⬇️ Descargar CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name=f"moi_salesrep_{pd.to_datetime(dstart_eff):%Y%m%d}_{pd.to_datetime(dend_eff):%Y%m%d}.csv",
    mime="text/csv",
)

# Leyenda de colores
st.caption("Leyenda MOI:")
st.markdown(
    "".join(
        f"<span style='display:inline-block;background:{col};color:{_contrast(col)};"
        f"padding:3px 8px;margin-right:6px;border-radius:6px;font-weight:600'>{name}</span>"
        for name, col in PALETA.items()
    ),
    unsafe_allow_html=True,
)

# ===================== Charts (con formatos) =====================

st.markdown("### 📊 Gráficos del período")

n_reps = len(g)
if n_reps == 0:
    st.info("No hay datos para graficar en este período.")
else:
    if n_reps <= 3:
        st.caption("Hay pocos reps en el período, mostrando todos (sin slider).")
        topN = n_reps
    else:
        topN = st.slider("Top reps por revenue", min_value=3, max_value=min(20, n_reps), value=min(10, n_reps))

    g_top = g.head(topN)

    # tooltips y ejes formateados
    t_rev   = alt.Tooltip('revenue_sum:Q', title='Revenue', format='$,.0f')
    t_prof  = alt.Tooltip('profit_sum:Q',  title='Profit',  format='$,.0f')
    t_pct   = alt.Tooltip('profit_pct:Q',  title='Profit %', format='.1%')
    t_aov   = alt.Tooltip('aov:Q',         title='AOV', format='$,.0f')
    t_ord   = alt.Tooltip('orders:Q',      title='Órdenes')

    if not g_top.empty:
        # 1) Revenue por rep
        chart_rev = alt.Chart(g_top).mark_bar().encode(
            x=alt.X('revenue_sum:Q', title='Revenue', axis=alt.Axis(format='$,.0f')),
            y=alt.Y('sales_rep:N', sort='-x', title='Sales Rep'),
            tooltip=['sales_rep', t_rev, t_prof, t_ord, t_pct, t_aov]
        ).properties(height=320, title="Revenue por Sales Rep (Top N)")
        st.altair_chart(chart_rev, use_container_width=True)

        # 2) Profit por rep
        chart_profit = alt.Chart(g_top).mark_bar().encode(
            x=alt.X('profit_sum:Q', title='Profit', axis=alt.Axis(format='$,.0f')),
            y=alt.Y('sales_rep:N', sort='-x', title=None),
            tooltip=['sales_rep', t_prof, t_pct, t_rev]
        ).properties(height=260, title="Profit por Sales Rep (Top N)")
        st.altair_chart(chart_profit, use_container_width=True)

        # 3) Profit % por rep
        chart_margin = alt.Chart(g_top).mark_bar().encode(
            x=alt.X('profit_pct:Q', title='Profit %', axis=alt.Axis(format='.1%')),
            y=alt.Y('sales_rep:N', sort='-x', title=None),
            color=alt.Color('MOI Overall:N',
                            scale=alt.Scale(domain=list(PALETA.keys()),
                                            range=list(PALETA.values())),
                            legend=alt.Legend(title="MOI")),
            tooltip=['sales_rep', t_pct, t_rev, t_prof, t_aov, t_ord, 'MOI Overall']
        ).properties(height=260, title="Profit % por Sales Rep (Top N)")
        st.altair_chart(chart_margin, use_container_width=True)
    else:
        st.info("No hay suficientes reps para gráficos comparativos.")

# 4) Línea diaria: revenue por día
daily = (
    df_f.assign(day=df_f[COL_DATE].dt.floor('D'))
        .groupby('day', as_index=False)
        .agg(revenue=(COL_REV, 'sum'),
             profit=(COL_PROF, 'sum'),
             orders=(COL_INV, pd.Series.nunique))
)
if not daily.empty:
    line_rev = alt.Chart(daily).mark_line(point=True).encode(
        x=alt.X('day:T', title='Fecha'),
        y=alt.Y('revenue:Q', title='Revenue diario', axis=alt.Axis(format='$,.0f')),
        tooltip=[alt.Tooltip('day:T', title='Fecha'),
                 alt.Tooltip('revenue:Q', title='Revenue', format='$,.0f'),
                 alt.Tooltip('profit:Q',  title='Profit',  format='$,.0f'),
                 alt.Tooltip('orders:Q',  title='Órdenes')]
    ).properties(height=300, title="Revenue diario")
    st.altair_chart(line_rev, use_container_width=True)

# 5) Stacked por día y rep
by_day_rep = (
    df_f.assign(day=df_f[COL_DATE].dt.floor('D'))
        .groupby(['day', 'sales_rep'], as_index=False)
        .agg(revenue=(COL_REV, 'sum'))
)
if not by_day_rep.empty:
    stack_day = alt.Chart(by_day_rep).mark_bar().encode(
        x=alt.X('day:T', title='Fecha'),
        y=alt.Y('revenue:Q', title='Revenue', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('sales_rep:N', title='Sales Rep'),
        tooltip=[alt.Tooltip('day:T', title='Fecha'),
                 'sales_rep', alt.Tooltip('revenue:Q', title='Revenue', format='$,.0f')]
    ).properties(height=320, title="Revenue por día y rep (apilado)")
    st.altair_chart(stack_day, use_container_width=True)

# 6) Scatter: AOV vs Profit % (tamaño = órdenes)
if not g.empty:
    scatter = alt.Chart(g).mark_circle().encode(
        x=alt.X('aov:Q', title='AOV', axis=alt.Axis(format='$,.0f')),
        y=alt.Y('profit_pct:Q', title='Profit %', axis=alt.Axis(format='.1%')),
        size=alt.Size('orders:Q', title='Órdenes', scale=alt.Scale(range=[50, 1200])),
        color=alt.Color('MOI Overall:N',
                        scale=alt.Scale(domain=list(PALETA.keys()),
                                        range=list(PALETA.values())),
                        legend=alt.Legend(title="MOI")),
        tooltip=['sales_rep', t_aov, t_pct, t_ord, t_rev, t_prof, 'MOI Overall']
    ).properties(height=340, title="AOV vs Profit % (tamaño = órdenes)")
    st.altair_chart(scatter, use_container_width=True)
