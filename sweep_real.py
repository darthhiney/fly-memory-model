"""
Similarity sweeps on the REAL hemibrain wiring, with SHUFFLED wiring as a control.

Runs the same four experiments as the toy sweeps:
    punishment (forward, synapse rule)   reward (forward, synapse rule)
    backward pairing (shock before odor) threshold rule (your original idea)
each at 11 similarity levels x 20 random odors, on real and on shuffled wiring.

Needs mb_real_model.py and the three CSVs from fetch_hemibrain.py in the same folder.
Run:  py sweep_real.py
Output: a summary table per experiment, plus real_<experiment>_<wiring>_summary.csv
        and ..._raw.csv files for your spreadsheet.

Compare REAL vs SHUFFLED: a difference bigger than the +/- spread means the fly's
specific wiring matters. Note the real wiring's untrained baseline is not 0 (about
-0.33), so use the SHIFT columns, not the raw scores.
"""
import csv
import numpy as np
import mb_real_model as m

SIMILARITIES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
N_TRIALS = 20
SEED = 1
EXPERIMENTS = {            # name: (dopamine, rule, order)
    "punishment": ("punishment", "synapse", "forward"),
    "reward":     ("reward", "synapse", "forward"),
    "backward":   ("punishment", "synapse", "backward"),
    "threshold":  ("punishment", "threshold", "forward"),
}


def one_trial(similarity, P, W0, scale, dopamine, rule, order):
    W = W0.copy()
    A = m.make_odor()
    B = m.make_odor(A, similarity)
    kc_A, kc_B = m.kc_response(A, P), m.kc_response(B, P)
    row = {"similarity": similarity, "kc_overlap": (kc_A * kc_B).sum() / kc_A.sum(),
           "A_before": m.behavior(A, P, W, scale), "B_before": m.behavior(B, P, W, scale)}
    m.train(A, P, W, dopamine, rule=rule, order=order)
    row["A_after"] = m.behavior(A, P, W, scale)
    row["B_after"] = m.behavior(B, P, W, scale)
    row["A_shift"] = row["A_after"] - row["A_before"]
    row["B_shift"] = row["B_after"] - row["B_before"]
    row["transfer"] = row["B_shift"] / row["A_shift"]
    return row


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sweep(exp, wiring, P, W0):
    dopamine, rule, order = EXPERIMENTS[exp]
    scale = m.avoid_scale(P, W0)
    rows = [one_trial(s, P, W0, scale, dopamine, rule, order)
            for s in SIMILARITIES for _ in range(N_TRIALS)]
    name = f"real_{exp}_{wiring}"
    save(rows, f"{name}_raw.csv")
    summary = []
    print(f"\n=== {exp}: {dopamine}, rule={rule}, order={order}, {wiring.upper()} wiring ===")
    print(f"{'similarity':>10} {'KC overlap':>14} {'A shift':>14} {'B shift':>14} {'transfer':>14}")
    for s in SIMILARITIES:
        r = [x for x in rows if x["similarity"] == s]
        st = {k: (np.mean([x[k] for x in r]), np.std([x[k] for x in r]))
              for k in ("kc_overlap", "A_shift", "B_shift", "transfer")}
        summary.append({"similarity": s, **{f"{k}_mean": v[0] for k, v in st.items()},
                        **{f"{k}_spread": v[1] for k, v in st.items()}})
        pct = lambda k: f"{st[k][0]:.0%} +/- {st[k][1]:.0%}"
        num = lambda k: f"{st[k][0]:.2f} +/- {st[k][1]:.2f}"
        print(f"{s:>10} {pct('kc_overlap'):>14} {num('A_shift'):>14} "
              f"{num('B_shift'):>14} {pct('transfer'):>14}")
    save(summary, f"{name}_summary.csv")


m.rng = np.random.default_rng(SEED)
shuffled_P, shuffled_W = m.shuffled(m.pn_to_kc, m.kc_to_mbon)
for exp in EXPERIMENTS:
    sweep(exp, "real", m.pn_to_kc, m.kc_to_mbon)
    sweep(exp, "shuffled", shuffled_P, shuffled_W)
print("\nSaved real_<experiment>_<wiring>_summary.csv and _raw.csv files")
