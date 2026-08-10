import json
import numpy as np
import streamlit as st

from data.data_loader import load_data
from engine.Vasieck_model import (
    calculate_shock,
    calculate_correlations,
    params_calculations,
    simulate_paths,
)
from engine.strategy_horizon import calculate_strategy_horizon
from simulation.simulate_strategy import simulate_strategy
from visualisation.calculate_stats import calculate_statistics
from visualisation.plot_chart import plot_fanchart

MAX_STRATEGIES = 4
CALIBRATION_CUTOFF = "2005-01-01"
NUM_SIM = 2000

st.set_page_config(page_title="Symulator Obligacji Skarbowych", layout="wide")

with open("bonds_config.json", "r", encoding="utf-8") as f:
    BONDS_CONFIG = json.load(f)

INDEX_LABELS = {
    "fixed": "Stałe oprocentowanie",
    "nbp": "Zmienne (stopa referencyjna NBP)",
    "cpi": "Indeksowane inflacją (CPI)",
}


def bond_summary_line(bond):
    years = bond["timeframe_months"] / 12
    horizon_txt = f"{bond['timeframe_months']} mies. (~{years:.1f} lat)"
    bonus_txt = f"{bond['bonus_rate']*100:.2f}% przez pierwsze {bond['bonus_length']} mies." if bond["is_bonus"] else "brak"
    return years, horizon_txt, bonus_txt


def strategy_label(bond_names):
    return " → ".join(bond_names)


def strategy_settings_label(reinvest, belka_tax_rate):
    tax_txt = f"podatek {belka_tax_rate*100:.0f}%" if belka_tax_rate > 0 else "bez podatku"
    reinvest_txt = "reinwestycja odsetek" if reinvest else "wypłata odsetek"
    return f"{tax_txt}, {reinvest_txt}"


def strategy_total_months(bond_names):
    return calculate_strategy_horizon([BONDS_CONFIG[n] for n in bond_names])


def strategy_segments(bond_names):
    """Cumulative month ranges for each bond in the sequence, for chart annotation."""
    segments = []
    cursor = 0
    for name in bond_names:
        months = BONDS_CONFIG[name]["timeframe_months"]
        segments.append({"name": name, "start": cursor, "end": cursor + months})
        cursor += months
    return segments


# ----------------------------------------------------------------------------
# Session state setup
# ----------------------------------------------------------------------------
if "current_build" not in st.session_state:
    st.session_state.current_build = []          # bonds being added to the strategy in progress
if "saved_strategies" not in st.session_state:
    # list of dicts: {"bonds": [names...], "reinvest": bool, "belka_tax_rate": float}
    st.session_state.saved_strategies = []
if "results" not in st.session_state:
    st.session_state.results = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


def add_to_build(name):
    st.session_state.current_build.append(name)


def undo_last():
    if st.session_state.current_build:
        st.session_state.current_build.pop()


def clear_build():
    st.session_state.current_build = []


def save_strategy():
    if not st.session_state.current_build:
        return
    if len(st.session_state.saved_strategies) >= MAX_STRATEGIES:
        return

    reinvest = st.session_state.get("build_reinvest_choice", "Tak, reinwestuj odsetki").startswith("Tak")
    belka_tax_rate = 0.19 if st.session_state.get("build_tax_choice", "19% (standardowa)").startswith("19") else 0.0

    st.session_state.saved_strategies.append({
        "bonds": list(st.session_state.current_build),
        "reinvest": reinvest,
        "belka_tax_rate": belka_tax_rate,
    })
    st.session_state.current_build = []
    st.session_state.results = None


def delete_strategy(idx):
    st.session_state.saved_strategies.pop(idx)
    st.session_state.results = None
    st.session_state.current_idx = 0


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("💰 Symulator Strategii Obligacji Skarbowych")
st.caption(
    "Beta MVP — symulacja Monte Carlo (model Vasicka skalibrowany na danych historycznych NBP / CPI). "
    "Wyniki to projekcje, a nie gwarancja przyszłych zwrotów."
)

st.divider()

# ----------------------------------------------------------------------------
# Step 1: Capital
# ----------------------------------------------------------------------------
st.subheader("1. Kwota inwestycji")
initial_capital = st.number_input(
    "Ile chcesz zainwestować (PLN)?",
    min_value=100.0,
    max_value=10_000_000.0,
    value=10000.0,
    step=500.0,
)

st.divider()

# ----------------------------------------------------------------------------
# Step 2: Build strategies (sequences of bonds + their own tax/reinvest settings)
# ----------------------------------------------------------------------------
st.subheader(f"2. Zbuduj strategie do porównania (maks. {MAX_STRATEGIES})")
st.caption(
    "Strategia to sekwencja obligacji ułożonych jedna po drugiej (np. 3M + 1Y + 12Y), "
    "wraz z własnymi ustawieniami reinwestycji i podatku. Dzięki temu możesz porównać np. "
    "tę samą sekwencję z podatkiem i bez podatku (np. konto emerytalne zwolnione z podatku Belki)."
)

build_col, saved_col = st.columns([3, 2])

with build_col:
    st.markdown("**Buduj bieżącą strategię**")

    bond_names = list(BONDS_CONFIG.keys())
    cols_per_row = 4
    rows = [bond_names[i:i + cols_per_row] for i in range(0, len(bond_names), cols_per_row)]

    build_disabled = len(st.session_state.saved_strategies) >= MAX_STRATEGIES

    for row in rows:
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            bond = BONDS_CONFIG[name]
            years, horizon_txt, bonus_txt = bond_summary_line(bond)

            with col:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.caption(INDEX_LABELS.get(bond["index_type"], bond["index_type"]))
                    st.write(f"⏳ {horizon_txt}")
                    st.write(f"🎁 {bonus_txt}")
                    st.write(f"➕ Marża: {bond['margin']*100:.2f}%")
                    st.button(
                        "➕ Dodaj do sekwencji",
                        key=f"btn_{name}",
                        use_container_width=True,
                        on_click=add_to_build,
                        args=(name,),
                        disabled=build_disabled,
                    )

    st.write("")
    current = st.session_state.current_build
    if current:
        total_m = strategy_total_months(current)
        st.info(f"**Bieżąca sekwencja:** {strategy_label(current)}  \n⏳ Łącznie: {total_m} mies. (~{total_m/12:.1f} lat)")
    else:
        st.caption("Bieżąca sekwencja jest pusta — dodaj przynajmniej jedną obligację.")

    b1, b2, b3 = st.columns(3)
    b1.button("↩️ Cofnij ostatnią", on_click=undo_last, use_container_width=True, disabled=not current)
    b2.button("🗑️ Wyczyść", on_click=clear_build, use_container_width=True, disabled=not current)

    st.write("")
    st.markdown("**Ustawienia tej strategii**")
    st.caption(
        "Te ustawienia zostaną zapisane razem z sekwencją — możesz zapisać tę samą sekwencję "
        "ponownie z innymi ustawieniami, aby porównać wyniki."
    )
    s1, s2 = st.columns(2)
    with s1:
        st.radio(
            "Czy reinwestować odsetki (dla obligacji niekapitalizujących)?",
            options=["Tak, reinwestuj odsetki", "Nie, wypłacaj odsetki na bieżąco"],
            index=0,
            key="build_reinvest_choice",
        )
    with s2:
        st.radio(
            "Stawka podatku Belki (od zysków kapitałowych)",
            options=["19% (standardowa)", "0% (np. konto zwolnione z podatku)"],
            index=0,
            key="build_tax_choice",
        )

    b3.button(
        "💾 Zapisz strategię",
        type="primary",
        use_container_width=True,
        on_click=save_strategy,
        disabled=(not current) or build_disabled,
    )
    if build_disabled:
        st.warning(f"Zapisano już maksymalną liczbę strategii ({MAX_STRATEGIES}). Usuń jedną, aby dodać nową.")

with saved_col:
    st.markdown(f"**Zapisane strategie ({len(st.session_state.saved_strategies)}/{MAX_STRATEGIES})**")
    if not st.session_state.saved_strategies:
        st.caption("Brak zapisanych strategii.")
    for i, strat in enumerate(st.session_state.saved_strategies):
        with st.container(border=True):
            total_m = strategy_total_months(strat["bonds"])
            st.write(f"**Strategia {i + 1}:** {strategy_label(strat['bonds'])}")
            st.caption(f"⏳ Łącznie: {total_m} mies. (~{total_m/12:.1f} lat)")
            st.caption(f"⚙️ {strategy_settings_label(strat['reinvest'], strat['belka_tax_rate'])}")
            st.button("✕ Usuń", key=f"del_{i}", on_click=delete_strategy, args=(i,))

st.divider()

# ----------------------------------------------------------------------------
# Step 3: Run
# ----------------------------------------------------------------------------
saved = st.session_state.saved_strategies
run = st.button("📊 Oblicz i porównaj strategie", type="primary", disabled=(len(saved) == 0))

if run and saved:
    with st.spinner("Kalibruję model i uruchamiam symulację Monte Carlo..."):
        try:
            matrix = load_data(["raw_nbp", "raw_cpi"], cutoff_date=CALIBRATION_CUTOFF)

            strategy_bond_lists = [[BONDS_CONFIG[n] for n in strat["bonds"]] for strat in saved]
            horizon = max(calculate_strategy_horizon(s) for s in strategy_bond_lists)

            np.random.seed(42)
            params_list = params_calculations(matrix)
            shocks = calculate_shock(matrix, horizon, NUM_SIM)
            stacked_shocks, _ = calculate_correlations(matrix, shocks)
            paths = simulate_paths(stacked_shocks, params_list, horizon, NUM_SIM)

            paths_mapping = {
                "nbp": paths[0],
                "cpi": paths[1],
                "fixed": paths[0],
            }

            results = []
            for strat, strat_bonds in zip(saved, strategy_bond_lists):
                global_matrix, _ = simulate_strategy(
                    strategy=strat_bonds,
                    initial_capital=initial_capital,
                    paths_mapping=paths_mapping,
                    num_sim=NUM_SIM,
                    belka_tax_rate=strat["belka_tax_rate"],
                    reinvest=strat["reinvest"],
                )
                stats = calculate_statistics(global_matrix, initial_capital)
                results.append({
                    "names": strat["bonds"],
                    "label": strategy_label(strat["bonds"]),
                    "settings_label": strategy_settings_label(strat["reinvest"], strat["belka_tax_rate"]),
                    "reinvest": strat["reinvest"],
                    "belka_tax_rate": strat["belka_tax_rate"],
                    "stats": stats,
                    "horizon": calculate_strategy_horizon(strat_bonds),
                    "segments": strategy_segments(strat["bonds"]),
                })

            st.session_state.results = results
            st.session_state.current_idx = 0
        except Exception as e:
            st.error(f"Wystąpił błąd podczas symulacji: {e}")
            st.session_state.results = None

st.divider()

# ----------------------------------------------------------------------------
# Step 4: Results carousel
# ----------------------------------------------------------------------------
if st.session_state.results:
    results = st.session_state.results
    n = len(results)
    idx = st.session_state.current_idx % n
    current_result = results[idx]

    nav_left, nav_title, nav_right = st.columns([1, 6, 1])
    with nav_left:
        if st.button("◀", use_container_width=True, disabled=(n <= 1)):
            st.session_state.current_idx = (idx - 1) % n
            st.rerun()
    with nav_right:
        if st.button("▶", use_container_width=True, disabled=(n <= 1)):
            st.session_state.current_idx = (idx + 1) % n
            st.rerun()
    with nav_title:
        st.markdown(
            f"<h3 style='text-align:center;margin:0;'>{current_result['label']}</h3>"
            f"<p style='text-align:center;color:gray;'>Strategia {idx + 1} z {n} &nbsp;•&nbsp; {current_result['settings_label']}</p>",
            unsafe_allow_html=True,
        )

    fig = plot_fanchart(
        current_result["stats"],
        current_result["horizon"],
        title=f"Projekcja kapitału — {current_result['label']}",
        segments=current_result["segments"],
    )
    st.pyplot(fig)
    seq_txt = "  →  ".join(f"**{seg['name']}** (mies. {seg['start']}–{seg['end']})" for seg in current_result["segments"])
    st.caption(f"Kolejność w strategii: {seq_txt}")
    st.caption(f"Ustawienia: {current_result['settings_label']}.")
    st.caption("Przerywane pionowe linie oznaczają moment, w którym kończy się jedna obligacja, a zaczyna kolejna w sekwencji.")

    summary = current_result["stats"]["summary"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Pesymistyczny wynik (5%)", f"{summary['worst_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['worst_profit']:,.0f} PLN".replace(",", " "))
    m2.metric("Średni oczekiwany wynik", f"{summary['mean_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['mean_profit']:,.0f} PLN".replace(",", " "))
    m3.metric("Optymistyczny wynik (95%)", f"{summary['best_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['best_profit']:,.0f} PLN".replace(",", " "))

    with st.expander("Szczegóły obligacji w tej strategii"):
        for seg in current_result["segments"]:
            b = BONDS_CONFIG[seg["name"]]
            years, horizon_txt, bonus_txt = bond_summary_line(b)
            st.markdown(f"**{seg['name']}** (miesiące {seg['start']}–{seg['end']})")
            st.write(f"- Typ oprocentowania: {INDEX_LABELS.get(b['index_type'], b['index_type'])}")
            st.write(f"- Okres: {horizon_txt}")
            st.write(f"- Bonus na start: {bonus_txt}")
            st.write(f"- Marża ponad indeks: {b['margin']*100:.2f}%")
            st.write(f"- Kapitalizacja odsetek co: {b['capitalisation_period']} mies.")
            st.write(f"- Kapitalizuje odsetki: {'Tak' if b['does_capitalise'] else 'Nie'}")
            st.write(f"- Kara za wcześniejszy wykup: {b['early_buyout_penalty']*100:.2f}%")

    st.divider()
    st.subheader("Porównanie wszystkich zapisanych strategii")
    comp_rows = []
    for r in results:
        s = r["stats"]["summary"]
        comp_rows.append({
            "Strategia": r["label"],
            "Ustawienia": r["settings_label"],
            "Okres (mies.)": r["horizon"],
            "Pesymistyczny (PLN)": round(s["worst_wealth"], 0),
            "Średni (PLN)": round(s["mean_wealth"], 0),
            "Optymistyczny (PLN)": round(s["best_wealth"], 0),
        })
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)
else:
    st.caption("Zbuduj i zapisz strategie, a następnie kliknij „Oblicz i porównaj strategie”, aby zobaczyć wyniki.")
