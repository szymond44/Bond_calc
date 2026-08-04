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


# ----------------------------------------------------------------------------
# Session state setup
# ----------------------------------------------------------------------------
if "selected_bonds" not in st.session_state:
    st.session_state.selected_bonds = []
if "results" not in st.session_state:
    st.session_state.results = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


def toggle_bond(name):
    sel = st.session_state.selected_bonds
    if name in sel:
        sel.remove(name)
    else:
        if len(sel) >= MAX_STRATEGIES:
            st.warning(f"Możesz porównać maksymalnie {MAX_STRATEGIES} strategie naraz. Odznacz jedną, aby dodać kolejną.")
        else:
            sel.append(name)


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
# Step 2: Bond blocks
# ----------------------------------------------------------------------------
st.subheader(f"2. Wybierz strategie do porównania (maks. {MAX_STRATEGIES})")
st.caption("Kliknij kartę obligacji, aby dodać ją do porównania.")

bond_names = list(BONDS_CONFIG.keys())
cols_per_row = 4
rows = [bond_names[i:i + cols_per_row] for i in range(0, len(bond_names), cols_per_row)]

for row in rows:
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        bond = BONDS_CONFIG[name]
        years, horizon_txt, bonus_txt = bond_summary_line(bond)
        is_selected = name in st.session_state.selected_bonds

        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(INDEX_LABELS.get(bond["index_type"], bond["index_type"]))
                st.write(f"⏳ Okres: {horizon_txt}")
                st.write(f"🎁 Bonus: {bonus_txt}")
                st.write(f"➕ Marża: {bond['margin']*100:.2f}%")
                st.write(f"🔄 Kapitalizacja: co {bond['capitalisation_period']} mies.")
                btn_label = "✅ Wybrano" if is_selected else "➕ Dodaj do porównania"
                st.button(
                    btn_label,
                    key=f"btn_{name}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                    on_click=toggle_bond,
                    args=(name,),
                )

selected = st.session_state.selected_bonds
if selected:
    st.success(f"Wybrane strategie ({len(selected)}/{MAX_STRATEGIES}): {', '.join(selected)}")
else:
    st.info("Nie wybrano jeszcze żadnej strategii.")

st.divider()

# ----------------------------------------------------------------------------
# Step 3: Reinvest & tax
# ----------------------------------------------------------------------------
st.subheader("3. Parametry inwestycji")
c1, c2 = st.columns(2)
with c1:
    reinvest_choice = st.radio(
        "Czy reinwestować odsetki (dla obligacji niekapitalizujących)?",
        options=["Tak, reinwestuj odsetki", "Nie, wypłacaj odsetki na bieżąco"],
        index=0,
    )
    reinvest = reinvest_choice.startswith("Tak")

with c2:
    belka_choice = st.radio(
        "Stawka podatku Belki (od zysków kapitałowych)",
        options=["19% (standardowa)", "0% (np. konto zwolnione z podatku)"],
        index=0,
    )
    belka_tax_rate = 0.19 if belka_choice.startswith("19") else 0.0

st.divider()

# ----------------------------------------------------------------------------
# Step 4: Run
# ----------------------------------------------------------------------------
run = st.button("📊 Oblicz i porównaj strategie", type="primary", disabled=(len(selected) == 0))

if run and selected:
    with st.spinner("Kalibruję model i uruchamiam symulację Monte Carlo..."):
        try:
            matrix = load_data(["raw_nbp", "raw_cpi"], cutoff_date=CALIBRATION_CUTOFF)

            strategies = [[BONDS_CONFIG[name]] for name in selected]
            horizon = max(sum(b["timeframe_months"] for b in s) for s in strategies)

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
            for name, strat in zip(selected, strategies):
                global_matrix, _ = simulate_strategy(
                    strategy=strat,
                    initial_capital=initial_capital,
                    paths_mapping=paths_mapping,
                    num_sim=NUM_SIM,
                    belka_tax_rate=belka_tax_rate,
                    reinvest=reinvest,
                )
                stats = calculate_statistics(global_matrix, initial_capital)
                results.append({
                    "name": name,
                    "bond": BONDS_CONFIG[name],
                    "stats": stats,
                    "horizon": strat[0]["timeframe_months"],
                })

            st.session_state.results = results
            st.session_state.current_idx = 0
        except Exception as e:
            st.error(f"Wystąpił błąd podczas symulacji: {e}")
            st.session_state.results = None

st.divider()

# ----------------------------------------------------------------------------
# Step 5: Results carousel
# ----------------------------------------------------------------------------
if st.session_state.results:
    results = st.session_state.results
    n = len(results)
    idx = st.session_state.current_idx % n
    current = results[idx]

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
            f"<h2 style='text-align:center;margin:0;'>{current['name']}</h2>"
            f"<p style='text-align:center;color:gray;'>Strategia {idx + 1} z {n}</p>",
            unsafe_allow_html=True,
        )

    fig = plot_fanchart(current["stats"], current["horizon"], title=f"Projekcja kapitału — {current['name']}")
    st.pyplot(fig)

    summary = current["stats"]["summary"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Pesymistyczny wynik (5%)", f"{summary['worst_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['worst_profit']:,.0f} PLN".replace(",", " "))
    m2.metric("Średni oczekiwany wynik", f"{summary['mean_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['mean_profit']:,.0f} PLN".replace(",", " "))
    m3.metric("Optymistyczny wynik (95%)", f"{summary['best_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['best_profit']:,.0f} PLN".replace(",", " "))

    with st.expander("Szczegóły produktu"):
        b = current["bond"]
        years, horizon_txt, bonus_txt = bond_summary_line(b)
        st.write(f"- **Typ oprocentowania:** {INDEX_LABELS.get(b['index_type'], b['index_type'])}")
        st.write(f"- **Okres:** {horizon_txt}")
        st.write(f"- **Bonus na start:** {bonus_txt}")
        st.write(f"- **Marża ponad indeks:** {b['margin']*100:.2f}%")
        st.write(f"- **Kapitalizacja odsetek co:** {b['capitalisation_period']} mies.")
        st.write(f"- **Kapitalizuje odsetki (dolicza do kapitału bez wypłaty):** {'Tak' if b['does_capitalise'] else 'Nie'}")
        st.write(f"- **Kara za wcześniejszy wykup:** {b['early_buyout_penalty']*100:.2f}%")

    # Quick side-by-side comparison table across all selected strategies
    st.divider()
    st.subheader("Porównanie wszystkich wybranych strategii")
    comp_rows = []
    for r in results:
        s = r["stats"]["summary"]
        comp_rows.append({
            "Strategia": r["name"],
            "Okres (mies.)": r["horizon"],
            "Pesymistyczny (PLN)": round(s["worst_wealth"], 0),
            "Średni (PLN)": round(s["mean_wealth"], 0),
            "Optymistyczny (PLN)": round(s["best_wealth"], 0),
        })
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)
else:
    st.caption("Wybierz strategie i kliknij „Oblicz i porównaj strategie”, aby zobaczyć wyniki.")
