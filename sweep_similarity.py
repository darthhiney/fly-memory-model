"""
Similarity sweep (generalization curve) for the toy model, with repeats.

For each similarity value it runs N_TRIALS independent trials. Each trial builds a
fresh random circuit and fresh random odors, trains odor A, and records everything.
Then it prints means +/- spread and saves the raw trials.

Run:  py sweep_similarity.py      (punishment, synapse rule, forward order)
The other experiments use the same code with different settings:
      py sweep_reward.py           (reward dopamine)
      py sweep_backward.py         (shock BEFORE odor)
      py sweep_threshold.py        (your threshold rule instead of synapses)

Output: printed summary table, plus <name>_raw.csv (every trial) and
        <name>_summary.csv (means and spreads), ready for a spreadsheet.
"""
import csv
import numpy as np
import mb_memory_model as m

SIMILARITIES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
N_TRIALS = 20
SEED = 1          # change this to check the results don't depend on the seed


def one_trial(similarity, dopamine, rule, order):
    circuit = m.build_random_circuit()
    gain = np.ones(m.N_MBON)
    A = m.make_odor()
    B = m.make_odor(A, similarity=similarity)
    kc_A, kc_B = m.kc_response(A, circuit[0]), m.kc_response(B, circuit[0])
    row = {"similarity": similarity, "kc_overlap": (kc_A * kc_B).sum() / kc_A.sum(),
           "A_before": m.behavior(A, circuit, gain), "B_before": m.behavior(B, circuit, gain)}
    m.train(A, circuit, gain, dopamine, rule=rule, order=order)
    row["A_after"] = m.behavior(A, circuit, gain)
    row["B_after"] = m.behavior(B, circuit, gain)
    row["A_shift"] = row["A_after"] - row["A_before"]
    row["B_shift"] = row["B_after"] - row["B_before"]
    row["transfer"] = row["B_shift"] / row["A_shift"]
    return row


def run_sweep(dopamine="punishment", rule="synapse", order="forward",
              name="similarity_sweep"):
    m.rng = np.random.default_rng(SEED)
    rows = [one_trial(s, dopamine, rule, order) for s in SIMILARITIES for _ in range(N_TRIALS)]
    with open(f"{name}_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    print(f"{dopamine} dopamine, rule={rule}, order={order}, "
          f"{N_TRIALS} trials per similarity (mean +/- spread)\n")
    print(f"{'similarity':>10} {'KC overlap':>14} {'A shift':>14} {'B shift':>14} {'transfer':>14}")
    for s in SIMILARITIES:
        r = [x for x in rows if x["similarity"] == s]
        stats = {k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r]))
                 for k in ("kc_overlap", "A_shift", "B_shift", "transfer")}
        summary.append({"similarity": s, **{f"{k}_mean": v[0] for k, v in stats.items()},
                        **{f"{k}_spread": v[1] for k, v in stats.items()}})
        fmt = lambda k, pct=False: (f"{stats[k][0]:.0%} +/- {stats[k][1]:.0%}" if pct
                                    else f"{stats[k][0]:.2f} +/- {stats[k][1]:.2f}")
        print(f"{s:>10} {fmt('kc_overlap', True):>14} {fmt('A_shift'):>14} "
              f"{fmt('B_shift'):>14} {fmt('transfer', True):>14}")

    with open(f"{name}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved {name}_raw.csv and {name}_summary.csv")


if __name__ == "__main__":
    run_sweep()
