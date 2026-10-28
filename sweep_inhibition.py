"""
Step 4: how fast does Dijkstra lose the memory as inhibition increases?

dijkstra_vs_sim.py found that with 30% inhibitory neurons Dijkstra misses most of
the learned memory (it can't use negative connections). This turns that one point
into a curve: 0%, 10%, 20%, 30%, 40% and 50% inhibitory L1 neurons, 100 trials each,
punishment learning with the synapse rule, similar odor B at similarity 0.6.
Each trial builds a fresh random circuit and fresh odors.

Run:  py sweep_inhibition.py      (takes a minute or two)
Needs mb_memory_model.py and dijkstra_vs_sim.py in the same folder.
Output: a table, plus inhibition_sweep_raw.csv and inhibition_sweep_summary.csv.
"""
import csv
import numpy as np
import mb_memory_model as m
import dijkstra_vs_sim as d

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # fraction of L1 neurons that are inhibitory
N_TRIALS = 100
SIMILARITY = 0.6
SEED = 1
REPORT = ["A_sim_shift", "A_dij_shift", "C_sim_shift", "C_dij_shift",
          "transfer_sim", "transfer_dij", "rank_agreement", "change_agreement"]


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


rows = []
for level in LEVELS:
    m.rng = np.random.default_rng(SEED)
    d.INHIBITORY = level
    for _ in range(N_TRIALS):
        rows.append({"inhibitory": level, **d.one_trial(similarity=SIMILARITY, rule="synapse")})
save(rows, "inhibition_sweep_raw.csv")

summary = []
print(f"=== punishment, synapse rule, similarity {SIMILARITY}, {N_TRIALS} trials per level (means) ===")
print(f"{'inhib':>6} {'A sim':>7} {'A dij':>7} {'captured':>9} {'C sim':>7} {'C dij':>7} "
      f"{'xfer sim':>9} {'xfer dij':>9} {'rank':>6} {'change':>7}")
for level in LEVELS:
    r = [x for x in rows if x["inhibitory"] == level]
    st = {k: (np.nanmean([x[k] for x in r]), np.nanstd([x[k] for x in r])) for k in REPORT}
    # share of the simulated memory that Dijkstra sees (100% = all of it), from the means
    captured = st["A_dij_shift"][0] / st["A_sim_shift"][0]
    summary.append({"inhibitory": level, "captured": captured,
                    **{f"{k}_mean": v[0] for k, v in st.items()},
                    **{f"{k}_spread": v[1] for k, v in st.items()}})
    print(f"{level:>6.0%} {st['A_sim_shift'][0]:>7.2f} {st['A_dij_shift'][0]:>7.2f} "
          f"{captured:>9.0%} {st['C_sim_shift'][0]:>7.2f} {st['C_dij_shift'][0]:>7.2f} "
          f"{st['transfer_sim'][0]:>9.0%} {st['transfer_dij'][0]:>9.0%} "
          f"{st['rank_agreement'][0]:>6.2f} {st['change_agreement'][0]:>7.2f}")
save(summary, "inhibition_sweep_summary.csv")
print("\nSaved inhibition_sweep_raw.csv and inhibition_sweep_summary.csv  (spreads are in the summary)")
