"""
Step 3: is the REAL wiring different from MANY shuffled wirings, not just one?

sweep_real.py compared the real hemibrain circuit with a single shuffle, so a
difference (or no difference) could be luck. This builds N_SHUFFLES shuffled
circuits and runs the same similarity sweep on each, with the SAME odors as the
real circuit, then asks where the real circuit falls among them.

Experiments: punishment and reward (forward, synapse rule).
Needs mb_real_model.py and neurons.csv, pn_kc.csv, kc_mbon.csv in the same folder.
Run:  py sweep_real_vs_shuffles.py      (takes several minutes)
Output: a table per experiment, plus shuffles_<experiment>_summary.csv (real vs the
        shuffled range at each similarity) and shuffles_<experiment>_curves.csv
        (every circuit's own curve, for plotting).
"""
import csv
import numpy as np
import mb_real_model as m

SIMILARITIES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
N_TRIALS = 20        # random odors per similarity, the same odors for every circuit
N_SHUFFLES = 100     # how many shuffled circuits to compare against
SEED = 1             # odors
SHUFFLE_SEED = 2     # which shuffles
EXPERIMENTS = {"punishment": "punishment", "reward": "reward"}


def shuffle(P, W, rng):
    """Same as mb_real_model.shuffled, with its own random generator so the odors stay fixed."""
    return (np.array([rng.permutation(row) for row in P]),
            np.array([rng.permutation(row) for row in W]))


def curve(P, W0, dopamine):
    """Mean baseline, KC overlap and transfer at each similarity, for one circuit."""
    m.rng = np.random.default_rng(SEED)      # same odors for every circuit
    scale = m.avoid_scale(P, W0)
    out = {}
    for s in SIMILARITIES:
        base, overlap, transfer = [], [], []
        for _ in range(N_TRIALS):
            W = W0.copy()
            A = m.make_odor()
            B = m.make_odor(A, s)
            kc_A, kc_B = m.kc_response(A, P), m.kc_response(B, P)
            a0, b0 = m.behavior(A, P, W, scale), m.behavior(B, P, W, scale)
            m.train(A, P, W, dopamine)
            a1, b1 = m.behavior(A, P, W, scale), m.behavior(B, P, W, scale)
            base.append(a0)
            overlap.append((kc_A * kc_B).sum() / kc_A.sum())
            transfer.append((b1 - b0) / (a1 - a0))
        out[s] = (np.mean(base), np.mean(overlap), np.mean(transfer))
    return out


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


rng = np.random.default_rng(SHUFFLE_SEED)
shuffles = [shuffle(m.pn_to_kc, m.kc_to_mbon, rng) for _ in range(N_SHUFFLES)]

for exp, dopamine in EXPERIMENTS.items():
    print(f"\nRunning {exp} on the real circuit and {N_SHUFFLES} shuffled circuits...")
    real = curve(m.pn_to_kc, m.kc_to_mbon, dopamine)
    shuf = [curve(P, W, dopamine) for P, W in shuffles]

    curves = [{"circuit": "real", "similarity": s, "baseline": v[0], "kc_overlap": v[1],
               "transfer": v[2]} for s, v in real.items()]
    for i, c in enumerate(shuf, 1):
        curves += [{"circuit": f"shuffle_{i}", "similarity": s, "baseline": v[0],
                    "kc_overlap": v[1], "transfer": v[2]} for s, v in c.items()]
    save(curves, f"shuffles_{exp}_curves.csv")

    summary = []
    print(f"=== {exp}: transfer to the similar odor, real vs {N_SHUFFLES} shuffles ===")
    print(f"{'sim':>5} {'real':>7} {'shuffled mean':>14} {'shuffled range':>16} {'real outside?':>14}")
    for s in SIMILARITIES:
        t = np.array([c[s][2] for c in shuf])
        outside = real[s][2] < t.min() or real[s][2] > t.max()
        summary.append({"similarity": s, "real_transfer": real[s][2],
                        "shuffled_mean": t.mean(), "shuffled_min": t.min(), "shuffled_max": t.max(),
                        "real_kc_overlap": real[s][1],
                        "shuffled_kc_overlap_mean": np.mean([c[s][1] for c in shuf]),
                        "real_outside_range": outside})
        print(f"{s:>5} {real[s][2]:>7.0%} {t.mean():>14.0%} "
              f"{f'{t.min():.0%} to {t.max():.0%}':>16} {'YES' if outside else 'no':>14}")
    save(summary, f"shuffles_{exp}_summary.csv")

    b = np.array([np.mean([v[0] for v in c.values()]) for c in shuf])
    rb = np.mean([v[0] for v in real.values()])
    print(f"Untrained baseline: real {rb:.3f}, shuffled {b.mean():.3f} "
          f"(range {b.min():.3f} to {b.max():.3f}), real outside range: "
          f"{'YES' if rb < b.min() or rb > b.max() else 'no'}")

print("\nSaved shuffles_<experiment>_summary.csv and shuffles_<experiment>_curves.csv")
