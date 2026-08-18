import copy
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
from visualisation.calculate_cash_erosion import calculate_cash_erosion_paths, cash_erosion_summary_at
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


def strategy_label(bonds):
    return " → ".join(b["name"] for b in bonds)


def strategy_settings_label(reinvest, belka_tax_rate):
    tax_txt = f"podatek {belka_tax_rate*100:.0f}%" if belka_tax_rate > 0 else "bez podatku"
    reinvest_txt = "reinwestycja odsetek" if reinvest else "wypłata odsetek"
    return f"{tax_txt}, {reinvest_txt}"


def strategy_total_months(bonds):
    return calculate_strategy_horizon(bonds)


def strategy_segments(bonds):
    segments = []
    cursor = 0
    for bond in bonds:
        months = bond["timeframe_months"]
        segments.append({"name": bond["name"], "start": cursor, "end": cursor + months})
        cursor += months
    return segments


def clip_segments_to_horizon(segments, horizon):
    clipped = []
    for seg in segments:
        if seg["start"] >= horizon:
            break
        clipped.append({
            "name": seg["name"],
            "start": seg["start"],
            "end": min(seg["end"], horizon),
        })
    return clipped


if "current_build" not in st.session_state:
    st.session_state.current_build = []
if "saved_strategies" not in st.session_state:
    st.session_state.saved_strategies = []
if "results" not in st.session_state:
    st.session_state.results = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


def add_to_build(name):
    bond = copy.deepcopy(BONDS_CONFIG[name])
    bond["name"] = name
    
    margin_key = f"{name}_margin"
    if margin_key in st.session_state:
        bond["margin"] = st.session_state[margin_key] / 100.0
        
    penalty_key = f"{name}_penalty"
    if penalty_key in st.session_state:
        bond["early_buyout_penalty"] = st.session_state[penalty_key] / 100.0
        
    if bond["is_bonus"]:
        bonus_key = f"{name}_bonus_rate"
        if bonus_key in st.session_state:
            bond["bonus_rate"] = st.session_state[bonus_key] / 100.0
            
    st.session_state.current_build.append(bond)


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
        "bonds": copy.deepcopy(st.session_state.current_build),
        "reinvest": reinvest,
        "belka_tax_rate": belka_tax_rate,
    })
    st.session_state.current_build = []


def delete_strategy(idx):
    st.session_state.saved_strategies.pop(idx)
    st.session_state.current_idx = 0

    if not st.session_state.saved_strategies:
        st.session_state.results = None


def reset_bond_settings():
    for key in list(st.session_state.keys()):
        if key.endswith("_margin") or key.endswith("_penalty") or key.endswith("_bonus_rate"):
            del st.session_state[key]


def run_simulation(saved, initial_capital, dca_amount, dca_duration_months, time_horizon_years, seed=42):
    with st.spinner("Kalibruję model i uruchamiam symulację Monte Carlo..."):
        try:
            matrix = load_data(["raw_nbp", "raw_cpi"], cutoff_date=CALIBRATION_CUTOFF)

            strategy_bond_lists = [copy.deepcopy(strat["bonds"]) for strat in saved]
            horizon = max(calculate_strategy_horizon(s) for s in strategy_bond_lists)

            np.random.seed(seed)
            params_list = params_calculations(matrix)
            shocks = calculate_shock(matrix, horizon, NUM_SIM)
            stacked_shocks, _ = calculate_correlations(matrix, shocks)
            paths = simulate_paths(stacked_shocks, params_list, horizon, NUM_SIM)

            paths_mapping = {
                "nbp": paths[0],
                "cpi": paths[1],
                "fixed": paths[0],
            }

            max_horizon_months = None if time_horizon_years is None else int(round(time_horizon_years * 12))

            results = []
            for strat, strat_bonds in zip(saved, strategy_bond_lists):
                global_matrix, strat_total_invested, effective_horizon, penalty_info = simulate_strategy(
                    strategy=strat_bonds,
                    initial_capital=initial_capital,
                    paths_mapping=paths_mapping,
                    num_sim=NUM_SIM,
                    belka_tax_rate=strat["belka_tax_rate"],
                    reinvest=strat["reinvest"],
                    max_horizon_months=max_horizon_months,
                    dca_amount=dca_amount,
                    dca_duration_months=dca_duration_months,
                )
                
                cash_erosion_paths = calculate_cash_erosion_paths(paths[1], strat_total_invested, horizon)
                
                strat_horizon = calculate_strategy_horizon(strat_bonds)
                segments = clip_segments_to_horizon(strategy_segments(strat["bonds"]), effective_horizon)

                stats = calculate_statistics(global_matrix, strat_total_invested)
                cash_summary = cash_erosion_summary_at(cash_erosion_paths, effective_horizon - 1, strat_total_invested)
                
                results.append({
                    "bonds": strat["bonds"],
                    "label": strategy_label(strat["bonds"]),
                    "settings_label": strategy_settings_label(strat["reinvest"], strat["belka_tax_rate"]),
                    "reinvest": strat["reinvest"],
                    "belka_tax_rate": strat["belka_tax_rate"],
                    "stats": stats,
                    "horizon": effective_horizon,
                    "full_horizon": strat_horizon,
                    "segments": segments,
                    "cash_erosion": cash_summary,
                    "penalty_info": penalty_info,
                    "total_invested": strat_total_invested,
                })

            st.session_state.results = results
            st.session_state.current_idx = 0
        except Exception as e:
            st.error(f"Wystąpił błąd podczas symulacji: {e}")
            st.session_state.results = None


st.markdown(
    "<h1 style='text-align:center;'>Symulator Strategii Obligacji Skarbowych</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Wyniki to projekcja i nie gwarantują przyszłych zysków.</p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown("<h3 style='text-align:center;'>1. Kwota inwestycji i horyzont</h3>", unsafe_allow_html=True)
cap_c1, cap_c2, cap_c3, cap_c4 = st.columns(4)
with cap_c1:
    initial_capital = st.number_input(
        "Kwota startowa (PLN)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=10000.0,
        step=500.0,
    )
with cap_c2:
    dca_amount = st.number_input(
        "Miesięczna dopłata (PLN)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=0.0,
        step=100.0,
    )
with cap_c3:
    dca_duration = st.number_input(
        "Czas dopłat (miesiące)",
        min_value=1,
        max_value=360,
        value=None,
        step=12,
        placeholder="Do końca strategii",
    )
with cap_c4:
    time_horizon_years = st.number_input(
        "Horyzont inwestycji (lata)",
        min_value=1,
        max_value=30,
        value=None,
        step=1,
        placeholder="Pełny okres obligacji",
        help="Jeśli wybrana obligacja lub sekwencja obligacji jest dłuższa niż ten horyzont, "
             "symulacja zostanie przycięta na tym horyzoncie i naliczona zostanie kara za "
             "wcześniejszy wykup obligacji, w której akurat trwałaby inwestycja.",
    )

st.divider()

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
            
            display_bond = copy.deepcopy(bond)
            if f"{name}_margin" in st.session_state:
                display_bond["margin"] = st.session_state[f"{name}_margin"] / 100.0
            if f"{name}_penalty" in st.session_state:
                display_bond["early_buyout_penalty"] = st.session_state[f"{name}_penalty"] / 100.0
            if bond["is_bonus"] and f"{name}_bonus_rate" in st.session_state:
                display_bond["bonus_rate"] = st.session_state[f"{name}_bonus_rate"] / 100.0

            years, horizon_txt, bonus_txt = bond_summary_line(display_bond)

            with col:
                with st.container(border=True):
                    head_col1, head_col2 = st.columns([3, 1])
                    with head_col1:
                        st.markdown(f"**{name}**")
                    with head_col2:
                        with st.popover("⚙️"):
                            st.markdown("**Własne warunki**")
                            st.number_input("Marża (%)", value=bond["margin"]*100, step=0.1, key=f"{name}_margin")
                            st.number_input("Kara za wcześniejszy wykup (%)", value=bond["early_buyout_penalty"]*100, step=0.1, key=f"{name}_penalty")
                            if bond["is_bonus"]:
                                st.number_input("Bonus w 1. okresie (%)", value=bond["bonus_rate"]*100, step=0.1, key=f"{name}_bonus_rate")

                    st.caption(INDEX_LABELS.get(display_bond["index_type"], display_bond["index_type"]))
                    st.write(f"{horizon_txt}")
                    st.write(f"{bonus_txt}")
                    st.write(f"Marża: {display_bond['margin']*100:.2f}%")
                    st.button(
                        "Dodaj do sekwencji",
                        key=f"btn_{name}",
                        use_container_width=True,
                        on_click=add_to_build,
                        args=(name,),
                        disabled=build_disabled,
                    )

    st.button("Zresetuj własne warunki obligacji", on_click=reset_bond_settings)

    st.write("")
    current = st.session_state.current_build
    if current:
        total_m = strategy_total_months(current)
        st.info(f"**Bieżąca sekwencja:** {strategy_label(current)}  \nŁącznie: {total_m} mies. (~{total_m/12:.1f} lat)")
    else:
        st.caption("Bieżąca sekwencja jest pusta. Dodaj przynajmniej jedną obligację.")

    b_left, b_mid, b_right = st.columns([1, 4, 1])
    with b_mid:
        b1, b2, b3 = st.columns(3)
        b1.button("Cofnij ostatnią", on_click=undo_last, use_container_width=True, disabled=not current)
        b2.button("Wyczyść", on_click=clear_build, use_container_width=True, disabled=not current)

    st.write("")
    st.markdown("**Ustawienia tej strategii**")
    st.caption(
        "Te ustawienia zostaną zapisane razem z sekwencją. Możesz zapisać tę samą sekwencję "
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
        "Zapisz strategię",
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
            st.caption(f"Łącznie: {total_m} mies. (~{total_m/12:.1f} lat)")
            
            custom_notes = []
            for b in strat["bonds"]:
                if b["margin"] != BONDS_CONFIG[b["name"]]["margin"]:
                    custom_notes.append(f"{b['name']} (marża {b['margin']*100:.2f}%)")
            if custom_notes:
                st.caption(f"Własne parametry: {', '.join(custom_notes)}")
                
            st.caption(f"{strategy_settings_label(strat['reinvest'], strat['belka_tax_rate'])}")
            st.button("Usuń", key=f"del_{i}", on_click=delete_strategy, args=(i,))

st.divider()

saved = st.session_state.saved_strategies

run_col, randomize_col = st.columns(2)
with run_col:
    run = st.button("Oblicz i porównaj strategie", type="primary", use_container_width=True, disabled=(len(saved) == 0))
with randomize_col:
    randomize = st.button("Losuj nowy scenariusz", use_container_width=True, disabled=(len(saved) == 0))

if run and saved:
    run_simulation(saved, initial_capital, dca_amount, dca_duration, time_horizon_years)
elif randomize and saved:
    run_simulation(saved, initial_capital, dca_amount, dca_duration, time_horizon_years, seed=None)

st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "„Oblicz i porównaj strategie” pokazuje wynik dla jednego, ustalonego scenariusza, więc możesz "
    "spokojnie porównywać różne strategie na tych samych warunkach. „Losuj nowy scenariusz” losuje "
    "inny, równie prawdopodobny przebieg przyszłości (inną inflację i stopy procentowe), żebyś zobaczył/a, "
    "jak bardzo wyniki mogą się różnić."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

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
        title=f"Projekcja kapitału: {current_result['label']}",
        segments=current_result["segments"],
    )
    st.pyplot(fig)
    seq_txt = "  →  ".join(f"**{seg['name']}** (mies. {seg['start']}–{seg['end']})" for seg in current_result["segments"])
    st.caption(f"Kolejność w strategii: {seq_txt}")
    st.caption(f"Ustawienia: {current_result['settings_label']}.")
    st.caption("Przerywane pionowe linie oznaczają moment, w którym kończy się jedna obligacja, a zaczyna kolejna w sekwencji.")

    if current_result["penalty_info"] is not None:
        pi = current_result["penalty_info"]
        st.warning(
            f"Wybrany horyzont inwestycji ({time_horizon_years} lat) jest krótszy niż pełna sekwencja "
            f"obligacji ({current_result['full_horizon']} mies.). Symulacja została przycięta na "
            f"{current_result['horizon']} mies., co oznacza wcześniejszy wykup obligacji {pi['bond']}. "
            f"Naliczono karę za wcześniejszy wykup w wysokości {pi['rate']*100:.2f}%."
        )
    elif current_result["horizon"] < current_result["full_horizon"]:
        st.info(
            f"Wybrany horyzont inwestycji ({time_horizon_years} lat) jest krótszy niż pełna sekwencja "
            f"obligacji ({current_result['full_horizon']} mies.). Symulacja została przycięta na "
            f"{current_result['horizon']} mies., dokładnie na końcu obligacji, więc kara za wcześniejszy "
            f"wykup nie została naliczona."
        )

    summary = current_result["stats"]["summary"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Pesymistyczny wynik (5%)", f"{summary['worst_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['worst_profit']:,.0f} PLN".replace(",", " "))
    m2.metric("Średni oczekiwany wynik", f"{summary['mean_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['mean_profit']:,.0f} PLN".replace(",", " "))
    m3.metric("Optymistyczny wynik (95%)", f"{summary['best_wealth']:,.0f} PLN".replace(",", " "),
               delta=f"{summary['best_profit']:,.0f} PLN".replace(",", " "))

    st.write("")
    st.markdown("**Ryzyko inflacyjne, co by było, gdybyś tego nie zainwestował?**")
    cash = current_result["cash_erosion"]
    st.caption(
        f"Realna wartość (siła nabywcza) sumy {current_result['total_invested']:,.0f} PLN wpłaconego kapitału, "
        f"gdyby leżał jako gotówka przez {current_result['horizon']} mies. (uproszczony rzut na start symulacji). "
        f"Na podstawie tych samych symulowanych ścieżek inflacji CPI, co powyższy fanchart.".replace(",", " ")
    )
    change_worst = cash["worst_real_value"] - current_result["total_invested"]
    change_mean = cash["mean_real_value"] - current_result["total_invested"]
    change_best = cash["best_real_value"] - current_result["total_invested"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Wysoka inflacja (5%)", f"{cash['worst_real_value']:,.0f} PLN".replace(",", " "),
        delta=f"{change_worst:,.0f} PLN siły nabywczej".replace(",", " "),
    )
    c2.metric(
        "Scenariusz średni", f"{cash['mean_real_value']:,.0f} PLN".replace(",", " "),
        delta=f"{change_mean:,.0f} PLN siły nabywczej".replace(",", " "),
    )
    c3.metric(
        "Niska inflacja (95%)", f"{cash['best_real_value']:,.0f} PLN".replace(",", " "),
        delta=f"{change_best:,.0f} PLN siły nabywczej".replace(",", " "),
    )
    advantage = summary["mean_wealth"] - cash["mean_real_value"]
    st.caption(
        f"W scenariuszu średnim ta strategia daje o **{advantage:,.0f} PLN** więcej niż realna "
        f"wartość tych samych pieniędzy trzymanych jako gotówka.".replace(",", " ")
    )

    with st.expander("Szczegóły obligacji w tej strategii"):
        for i, b in enumerate(current_result["bonds"]):
            seg = current_result["segments"][i]
            years, horizon_txt, bonus_txt = bond_summary_line(b)
            st.markdown(f"**{b['name']}** (miesiące {seg['start']}–{seg['end']})")
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
        cash = r["cash_erosion"]
        comp_rows.append({
            "Strategia": r["label"],
            "Ustawienia": r["settings_label"],
            "Okres (mies.)": r["horizon"],
            "Wpłacono (PLN)": round(r["total_invested"], 0),
            "Pesymistyczny (PLN)": round(s["worst_wealth"], 0),
            "Średni (PLN)": round(s["mean_wealth"], 0),
            "Optymistyczny (PLN)": round(s["best_wealth"], 0),
            "Gotówka realnie (średnio, PLN)": round(cash["mean_real_value"], 0),
            "Przewaga nad gotówką (PLN)": round(s["mean_wealth"] - cash["mean_real_value"], 0),
        })
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)
else:
    st.markdown(
        "<p style='text-align:center;color:gray;'>Zbuduj i zapisz strategie, a następnie kliknij "
        "„Oblicz i porównaj strategie”, aby zobaczyć wyniki.</p>",
        unsafe_allow_html=True,
    )