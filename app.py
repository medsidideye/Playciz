# colle ici TOUT le code Streamlit que je t’ai donné
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="Playciz KPI Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def safe_to_datetime(s):
    """Convert string to datetime, coercing errors."""
    return pd.to_datetime(s, errors="coerce")

def build_start_datetime(df):
    """Build start_datetime from date+time if not already present."""
    if "start_datetime" in df.columns:
        dt = safe_to_datetime(df["start_datetime"])
        if dt.notna().any():
            df["start_datetime"] = dt
            return df

    # Otherwise, build from date and time
    if {"date", "time"} <= set(df.columns):
        df["start_datetime"] = safe_to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        raise ValueError("Le fichier matches doit contenir soit start_datetime, soit les colonnes date et time.")
    return df

def deduplicate_matches(df):
    """
    If matches contains multiple snapshots per match_id, keep one row per match_id.
    Rule: keep row with minimum hours_before if exists (closest to match time).
    """
    if "hours_before" in df.columns:
        df = df.sort_values(["match_id", "hours_before"]).groupby("match_id", as_index=False).first()
    else:
        df = df.drop_duplicates(subset=["match_id"])
    return df

def compute_revenue(df):
    """
    Estimate revenue using nb_members/nb_externals and final prices.
    If discount_percent & advantage exist -> apply discount when advantage==1.
    """
    df = df.copy()

    # Default: no discount
    if "discount_percent" not in df.columns:
        df["discount_percent"] = 0
    if "advantage" not in df.columns:
        df["advantage"] = 0

    discount = np.where(df["advantage"].astype(int) == 1, df["discount_percent"].astype(float), 0.0)
    discount_factor = 1.0 - (discount / 100.0)

    # Final prices after discount
    df["final_price_member"] = df["price_member"].astype(float) * discount_factor
    df["final_price_external"] = df["price_external"].astype(float) * discount_factor

    # Revenue
    df["revenue_est"] = (
        df["nb_members"].astype(int) * df["final_price_member"]
        + df["nb_externals"].astype(int) * df["final_price_external"]
    )

    # If match is canceled, you can decide revenue=0 (often true)
    if "status" in df.columns:
        df.loc[df["status"].astype(str).str.lower() == "canceled", "revenue_est"] = 0.0

    return df

def compute_late_cancel_rate(df):
    """
    Late cancel rate = cancellations <6h / total cancellations.
    canceled_at is string 'Not cancelled' or datetime string.
    """
    df = df.copy()
    if "status" not in df.columns or "canceled_at" not in df.columns:
        return np.nan

    df["status_l"] = df["status"].astype(str).str.lower()
    canceled = df[df["status_l"] == "canceled"].copy()
    if canceled.empty:
        return 0.0

    # Parse canceled_at
    canceled_at = canceled["canceled_at"].replace("Not cancelled", np.nan)
    canceled["canceled_at_dt"] = safe_to_datetime(canceled_at)
    canceled["start_dt"] = safe_to_datetime(canceled["start_datetime"])

    canceled["delta_hours"] = (canceled["start_dt"] - canceled["canceled_at_dt"]).dt.total_seconds() / 3600.0
    late = canceled["delta_hours"].notna() & (canceled["delta_hours"] < 6.0) & (canceled["delta_hours"] >= 0.0)

    return late.mean()

def equity_std_matches_7d(matches_df, mp_df, end_dt):
    """
    Equity = std dev of number of matches per member in last 7 days.
    Uses match_players table (member players only) + matches start_datetime.
    """
    if end_dt is None or pd.isna(end_dt):
        return np.nan

    start_window = end_dt - timedelta(days=7)

    # Filter played matches in window
    m = matches_df.copy()
    if "status" in m.columns:
        m = m[m["status"].astype(str).str.lower() == "played"]

    m = m[(m["start_datetime"] >= start_window) & (m["start_datetime"] <= end_dt)]
    if m.empty:
        return 0.0

    # Join with match_players and keep members
    mp = mp_df.copy()
    mp = mp[mp["is_member"].astype(int) == 1]
    joined = mp.merge(m[["match_id"]], on="match_id", how="inner")

    if joined.empty:
        return 0.0

    counts = joined.groupby("player_id")["match_id"].nunique()
    return float(counts.std(ddof=0))  # population std

def member_play_hours_per_month(matches_df, mp_df):
    """
    KPI: total member play time (hours/month).
    We compute "player-hours": each member in a match contributes duration.
    """
    m = matches_df.copy()
    if "status" in m.columns:
        m = m[m["status"].astype(str).str.lower() == "played"]

    if m.empty:
        return pd.DataFrame(columns=["month", "member_hours"])

    m["month"] = m["start_datetime"].dt.to_period("M").astype(str)

    mp = mp_df.copy()
    mp = mp[mp["is_member"].astype(int) == 1]

    joined = mp.merge(m[["match_id", "month", "duration_min"]], on="match_id", how="inner")
    joined["hours"] = joined["duration_min"].astype(float) / 60.0

    agg = joined.groupby("month")["hours"].sum().reset_index()
    agg.rename(columns={"hours": "member_hours"}, inplace=True)
    return agg.sort_values("month")

# -----------------------------
# Load data
# -----------------------------
st.title("Playciz — KPI Dashboard")

with st.sidebar:
    st.header("Chargement des données")
    matches_file = st.text_input("Fichier matches", value="matches_with_data_completed.csv")
    mp_file = st.text_input("Fichier match_players", value="match_players_with_feedback.csv")
    st.caption("Assure-toi que ces fichiers sont dans le même dossier que app.py (ou mets le chemin complet).")

@st.cache_data
def load_data(matches_file, mp_file):
    matches = pd.read_csv(matches_file)
    mp = pd.read_csv(mp_file)
    return matches, mp

try:
    matches_raw, mp_raw = load_data(matches_file, mp_file)
except Exception as e:
    st.error(f"Erreur chargement fichiers: {e}")
    st.stop()

# Prepare matches
matches = matches_raw.copy()
matches = deduplicate_matches(matches)
matches = build_start_datetime(matches)

# Basic columns check
needed_matches = {"match_id","sport","terrain_id","duration_min","is_full_final","status","canceled_at","nb_members","nb_externals","price_member","price_external","start_datetime"}
missing = needed_matches - set(matches.columns)
if missing:
    st.warning(f"Colonnes manquantes dans matches: {missing}. "
               f"Le dashboard fonctionnera partiellement. Ajoute-les si possible.")

# Prepare match_players
mp = mp_raw.copy()
needed_mp = {"match_id","player_id","is_member","feedback_score"}
missing_mp = needed_mp - set(mp.columns)
if missing_mp:
    st.warning(f"Colonnes manquantes dans match_players: {missing_mp}. "
               f"Équité / Satisfaction pourront être incomplètes.")

# Compute revenue
if {"nb_members","nb_externals","price_member","price_external"} <= set(matches.columns):
    matches = compute_revenue(matches)

# -----------------------------
# Filters
# -----------------------------
with st.sidebar:
    st.header("Filtres")

    # Date range
    min_dt = matches["start_datetime"].min()
    max_dt = matches["start_datetime"].max()
    date_range = st.date_input("Période", value=(min_dt.date(), max_dt.date()))

    # Sport / terrain
    sports = ["All"] + sorted(matches["sport"].dropna().astype(str).unique().tolist()) if "sport" in matches.columns else ["All"]
    sport_sel = st.selectbox("Sport", sports)

    terrains = ["All"] + sorted(matches["terrain_id"].dropna().astype(str).unique().tolist()) if "terrain_id" in matches.columns else ["All"]
    terrain_sel = st.selectbox("Terrain", terrains)

    status_opts = ["All"] + sorted(matches["status"].dropna().astype(str).unique().tolist()) if "status" in matches.columns else ["All"]
    status_sel = st.selectbox("Status", status_opts)

# Apply filters
f = matches.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    f = f[(f["start_datetime"].dt.date >= start_d) & (f["start_datetime"].dt.date <= end_d)]

if sport_sel != "All" and "sport" in f.columns:
    f = f[f["sport"].astype(str) == sport_sel]

if terrain_sel != "All" and "terrain_id" in f.columns:
    f = f[f["terrain_id"].astype(str) == terrain_sel]

if status_sel != "All" and "status" in f.columns:
    f = f[f["status"].astype(str) == status_sel]

# For join-based KPIs
mp_f = mp.merge(f[["match_id"]], on="match_id", how="inner") if "match_id" in f.columns else mp.copy()

# -----------------------------
# KPI Computations
# -----------------------------
# Fill rate
if "is_full_final" in f.columns:
    fill_rate = float((f["is_full_final"].astype(int) == 1).mean())
else:
    fill_rate = np.nan

# Revenue
revenue = float(f["revenue_est"].sum()) if "revenue_est" in f.columns else np.nan

# Satisfaction
if "feedback_score" in mp_f.columns:
    # only played matches
    played_ids = set(f.loc[f["status"].astype(str).str.lower() == "played", "match_id"]) if "status" in f.columns else set(f["match_id"])
    mp_played = mp_f[mp_f["match_id"].isin(played_ids)]
    sat = mp_played.loc[mp_played["feedback_score"].astype(float) > 0, "feedback_score"].astype(float).mean()
    satisfaction = float(sat) if not np.isnan(sat) else np.nan
else:
    satisfaction = np.nan

# Late cancel rate
late_cancel = compute_late_cancel_rate(f) if {"status","canceled_at"} <= set(f.columns) else np.nan

# Equity (7 days, end date = max date in filtered range)
end_dt = f["start_datetime"].max() if "start_datetime" in f.columns and not f.empty else None
equity_std = equity_std_matches_7d(f, mp_f, end_dt) if {"player_id","is_member"} <= set(mp_f.columns) else np.nan

# Member hours per month (player-hours)
member_hours_month = member_play_hours_per_month(f, mp_f) if {"duration_min"} <= set(f.columns) else pd.DataFrame()

# -----------------------------
# Display KPI cards
# -----------------------------
st.subheader("KPI — Résumé")
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Taux de remplissage", f"{fill_rate*100:.1f}%" if not np.isnan(fill_rate) else "NA")
c2.metric("Revenu estimé", f"{revenue:,.2f} €" if not np.isnan(revenue) else "NA")
c3.metric("Satisfaction moyenne", f"{satisfaction:.2f}/5" if not np.isnan(satisfaction) else "NA")
c4.metric("Taux annulation tardive (<6h)", f"{late_cancel*100:.1f}%" if not np.isnan(late_cancel) else "NA")
c5.metric("Équité (σ matchs/adhérent, 7j)", f"{equity_std:.2f}" if not np.isnan(equity_std) else "NA")

# Total member hours in filtered period
if not member_hours_month.empty:
    total_member_hours = float(member_hours_month["member_hours"].sum())
    c6.metric("Temps de jeu adhérents (h)", f"{total_member_hours:.1f}")
else:
    c6.metric("Temps de jeu adhérents (h)", "NA")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Évolution temporelle")

# Revenue over time (daily)
if "revenue_est" in f.columns and "start_datetime" in f.columns and not f.empty:
    rev_daily = f.copy()
    rev_daily["day"] = rev_daily["start_datetime"].dt.date
    rev_daily = rev_daily.groupby("day")["revenue_est"].sum().reset_index()

    st.write("Revenu par jour")
    st.line_chart(rev_daily.set_index("day"))

# Fill rate over time (weekly)
if "is_full_final" in f.columns and "start_datetime" in f.columns and not f.empty:
    fill_week = f.copy()
    fill_week["week"] = f["start_datetime"].dt.to_period("W").astype(str)
    fill_week = fill_week.groupby("week")["is_full_final"].mean().reset_index()

    st.write("Taux de remplissage par semaine")
    st.line_chart(fill_week.set_index("week"))

st.subheader("Joueurs, Équité, Satisfaction")

# Member hours per month
if not member_hours_month.empty:
    st.write("Temps de jeu adhérents (heures-joueur) par mois")
    st.bar_chart(member_hours_month.set_index("month"))

# Matches per member last 7 days (distribution)
if {"player_id","is_member"} <= set(mp_f.columns) and not f.empty:
    # distribution in the filtered period
    mp_members = mp_f[mp_f["is_member"].astype(int) == 1]
    counts_all = mp_members.groupby("player_id")["match_id"].nunique()
    if not counts_all.empty:
        st.write("Distribution: nombre de matchs par adhérent (période filtrée)")
        hist = np.histogram(counts_all.values, bins=range(0, int(counts_all.max()) + 2))
        hist_df = pd.DataFrame({"matches": hist[1][:-1], "count_players": hist[0]})
        st.bar_chart(hist_df.set_index("matches"))

# Satisfaction by member/external
if "feedback_score" in mp_f.columns and "is_member" in mp_f.columns:
    mp_sc = mp_f.copy()
    mp_sc["feedback_score"] = mp_sc["feedback_score"].astype(float)
    mp_sc = mp_sc[mp_sc["feedback_score"] > 0]
    if not mp_sc.empty:
        sat_by_type = mp_sc.groupby("is_member")["feedback_score"].mean().reset_index()
        sat_by_type["type"] = sat_by_type["is_member"].map({1: "Member", 0: "External"})
        st.write("Satisfaction moyenne — Membres vs Externes")
        st.bar_chart(sat_by_type.set_index("type")[["feedback_score"]])

st.subheader("Annulations")

if {"status","canceled_at","start_datetime"} <= set(f.columns) and not f.empty:
    canc = f[f["status"].astype(str).str.lower() == "canceled"].copy()
    if not canc.empty:
        canc_at = canc["canceled_at"].replace("Not cancelled", np.nan)
        canc["canceled_at_dt"] = safe_to_datetime(canc_at)
        canc["delta_hours"] = (canc["start_datetime"] - canc["canceled_at_dt"]).dt.total_seconds() / 3600.0
        canc["late_cancel"] = (canc["delta_hours"] < 6.0) & (canc["delta_hours"] >= 0.0)

        st.write("Répartition des délais d'annulation (heures avant match)")
        # simple histogram
        valid = canc["delta_hours"].dropna()
        if not valid.empty:
            bins = [0, 1, 3, 6, 12, 24, 48, 168]
            labels = ["<1h","1-3h","3-6h","6-12h","12-24h","24-48h","2-7j"]
            cats = pd.cut(valid, bins=bins, labels=labels, include_lowest=True, right=False)
            grp = cats.value_counts().sort_index()
            st.bar_chart(grp)

# -----------------------------
# Data preview
# -----------------------------
with st.expander("Aperçu des données (matches filtrés)"):
    st.dataframe(f.head(200))

with st.expander("Aperçu des données (match_players filtrés)"):
    st.dataframe(mp_f.head(200))
