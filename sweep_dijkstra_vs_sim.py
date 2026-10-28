"""
Sweep: Dijkstra vs simulation, 20 trials per similarity level, 4 conditions:
    synapse rule  x  inhibition on (30% of L1 neurons inhibitory) / off (0%)
    threshold rule x inhibition on / off
Each trial builds a fresh random circuit and fresh odors.

Run:  py sweep_dijkstra_vs_sim.py      (takes a minute or two)
Needs mb_memory_model.py and dijkstra_vs_sim.py in the same folder.
Output: a summary table per condition, plus dvs_<rule>_inhib<N>_raw.csv and
        dvs_<rule>_inhib<N>_summary.csv for your spreadsheet.
"""
import csv
import numpy as np
import mb_memory_model as m
import dijkstra_vs_sim as d

SIMILARITIES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
N_TRIALS = 20
SEED = 1
CONDITIONS = [("synapse", 0.0), ("synapse", 0.3), ("threshold", 0.0), ("threshold", 0.3)]
REPORT = ["A_sim_shift", "A_dij_shift", "C_sim_shift", "C_dij_shift",
          "transfer_sim", "transfer_dij", "rank_agreement", "change_agreement"]


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


for rule, inhib in CONDITIONS:
    m.rng = np.random.default_rng(SEED)
    d.INHIBITORY = inhib
    rows = []
    for s in SIMILARITIES:
        for _ in range(N_TRIALS):
            rows.append(d.one_trial(similarity=s, rule=rule))
    name = f"dvs_{rule}_inhib{int(inhib * 100)}"
    save(rows, f"{name}_raw.csv")

    summary = []
    print(f"\n=== rule={rule}, inhibitory={inhib:.0%}, {N_TRIALS} trials each (means) ===")
    print(f"{'sim':>5} {'A sim':>7} {'A dij':>7} {'C sim':>7} {'C dij':>7} "
          f"{'xfer sim':>9} {'xfer dij':>9} {'rank':>6} {'change':>7}")
    for s in SIMILARITIES:
        r = [x for x in rows if x["similarity"] == s]
        st = {k: (np.nanmean([x[k] for x in r]), np.nanstd([x[k] for x in r])) for k in REPORT}
        summary.append({"similarity": s, **{f"{k}_mean": v[0] for k, v in st.items()},
                        **{f"{k}_spread": v[1] for k, v in st.items()}})
        print(f"{s:>5} {st['A_sim_shift'][0]:>7.2f} {st['A_dij_shift'][0]:>7.2f} "
              f"{st['C_sim_shift'][0]:>7.2f} {st['C_dij_shift'][0]:>7.2f} "
              f"{st['transfer_sim'][0]:>9.0%} {st['transfer_dij'][0]:>9.0%} "
              f"{st['rank_agreement'][0]:>6.2f} {st['change_agreement'][0]:>7.2f}")
    save(summary, f"{name}_summary.csv")
print("\nSaved dvs_*_raw.csv and dvs_*_summary.csv  (spreads are in the summary CSVs)")
