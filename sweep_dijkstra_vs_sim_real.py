"""
Sweep: Dijkstra vs simulation on REAL wiring, 20 random odors per similarity level,
3 conditions:
    punishment (forward, synapse rule)   reward (forward, synapse rule)
    punishment, threshold rule (your original idea)

Run:  py sweep_dijkstra_vs_sim_real.py
Needs dijkstra_vs_sim_real.py and everything it needs in the same folder.
Output: a summary table per condition, plus dvsreal_<condition>_raw.csv and
        dvsreal_<condition>_summary.csv for your spreadsheet.
"""
import csv
import numpy as np
import mb_real_model as m
import dijkstra_vs_sim_real as d

SIMILARITIES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
N_TRIALS = 20
SEED = 1
CONDITIONS = {             # name: (dopamine, rule)
    "punishment": ("punishment", "synapse"),
    "reward":     ("reward", "synapse"),
    "threshold":  ("punishment", "threshold"),
}
REPORT = ["kc_overlap", "A_sim_change", "A_dij_change", "transfer_sim", "transfer_dij",
          "C_sim_spill", "C_dij_spill", "rank_agreement", "change_agreement"]


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


for name, (dopamine, rule) in CONDITIONS.items():
    m.rng = np.random.default_rng(SEED)
    rows = [d.one_trial(similarity=s, dopamine=dopamine, rule=rule)
            for s in SIMILARITIES for _ in range(N_TRIALS)]
    save(rows, f"dvsreal_{name}_raw.csv")

    summary = []
    print(f"\n=== {name} ({rule} rule), real wiring, {N_TRIALS} odors each (means) ===")
    print(f"{'sim':>5} {'overlap':>8} {'A sim':>7} {'A dij':>7} {'xfer sim':>9} {'xfer dij':>9} "
          f"{'C sim':>6} {'C dij':>6} {'rank':>6} {'change':>7}")
    for s in SIMILARITIES:
        r = [x for x in rows if x["similarity"] == s]
        st = {k: (np.nanmean([x[k] for x in r]), np.nanstd([x[k] for x in r])) for k in REPORT}
        summary.append({"similarity": s, **{f"{k}_mean": v[0] for k, v in st.items()},
                        **{f"{k}_spread": v[1] for k, v in st.items()}})
        print(f"{s:>5} {st['kc_overlap'][0]:>8.0%} {st['A_sim_change'][0]:>7.1%} "
              f"{st['A_dij_change'][0]:>7.1%} {st['transfer_sim'][0]:>9.0%} "
              f"{st['transfer_dij'][0]:>9.0%} {st['C_sim_spill'][0]:>6.0%} "
              f"{st['C_dij_spill'][0]:>6.0%} {st['rank_agreement'][0]:>6.2f} "
              f"{st['change_agreement'][0]:>7.2f}")
    save(summary, f"dvsreal_{name}_summary.csv")
print("\nSaved dvsreal_*_raw.csv and dvsreal_*_summary.csv  (spreads are in the summary CSVs)")
