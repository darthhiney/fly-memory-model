"""
Dijkstra vs simulation on REAL wiring: the real hemibrain mushroom body plus the real
neurons two steps downstream of the MBONs.

    odor -> KCs -> MBONs -> L1 (real MBON targets) -> L2 (real targets of L1)

Same question as dijkstra_vs_sim.py (the toy): how much of a learned memory does a
cheap shortest-path map capture, compared with simulating activity, and does it
falsely spread the memory to odors that weren't trained?

Differences from the toy:
  - There's no "approach / avoid" output: real downstream neurons don't come labeled.
    So everything is measured on the downstream neurons themselves (see the guide).
  - All connections are treated as EXCITATORY in both methods. The hemibrain doesn't
    say reliably which downstream neurons inhibit, so this version tests only the
    "one best path" failure, not the inhibition failure.
  - Connections from L1 back to L1 (loops) are left out, so activity flows one way.

Needs, in the same folder: mb_real_model.py, neurons.csv, pn_kc.csv, kc_mbon.csv
(fetch_hemibrain.py) and mbon_l1.csv, l1_l2.csv, down_neurons.csv (fetch_downstream.py).
Run:  py dijkstra_vs_sim_real.py
"""
import numpy as np
import pandas as pd
import networkx as nx
import mb_real_model as m

FIRE_THRESHOLD = 0.05   # simulation: activity below this (relative) doesn't pass on

# ---- Load the downstream wiring --------------------------------------------
mbon_l1 = pd.read_csv("mbon_l1.csv")
l1_l2 = pd.read_csv("l1_l2.csv")
type_of = dict(pd.read_csv("down_neurons.csv")[["bodyId", "type"]].itertuples(index=False))

mbon_l1 = mbon_l1[mbon_l1["bodyId_pre"].isin(m.mbon_index)]
l1_ids = np.sort(mbon_l1["bodyId_post"].unique())
l1_l2 = l1_l2[l1_l2["bodyId_pre"].isin(set(l1_ids))]
l2_ids = np.sort(l1_l2["bodyId_post"].unique())
i1 = {b: i for i, b in enumerate(l1_ids)}
i2 = {b: i for i, b in enumerate(l2_ids)}
down_ids = np.concatenate([l1_ids, l2_ids])
down_layer = np.array(["L1"] * len(l1_ids) + ["L2"] * len(l2_ids))

W1 = np.zeros((len(l1_ids), len(m.mbon_ids)))      # MBON -> L1, synapse counts
for pre, post, w in mbon_l1[["bodyId_pre", "bodyId_post", "weight"]].itertuples(index=False):
    W1[i1[post], m.mbon_index[pre]] += w
W2 = np.zeros((len(l2_ids), len(l1_ids)))          # L1 -> L2
for pre, post, w in l1_l2[["bodyId_pre", "bodyId_post", "weight"]].itertuples(index=False):
    W2[i2[post], i1[pre]] += w
print(f"Downstream: {len(l1_ids)} L1 and {len(l2_ids)} L2 neurons, "
      f"{int((W1 > 0).sum() + (W2 > 0).sum())} connections")

# ---- Simulation --------------------------------------------------------------
# Each neuron's input = weighted average of its inputs' activity (weights = share of
# its synapses), minus a small threshold. MBON activity is in units of the average
# MBON's response to a random odor on the untrained wiring, so learning shows up.
IN1 = W1 / W1.sum(axis=1, keepdims=True)
IN2 = W2 / W2.sum(axis=1, keepdims=True)
REF = np.mean([(m.kc_to_mbon @ m.kc_response(m.make_odor(), m.pn_to_kc)).mean()
               for _ in range(50)])


def simulate(odor, P, W):
    mbon = (W @ m.kc_response(odor, P)) / REF
    l1 = np.maximum(IN1 @ mbon - FIRE_THRESHOLD, 0)
    l2 = np.maximum(IN2 @ l1 - FIRE_THRESHOLD, 0)
    return np.concatenate([l1, l2])


# ---- Dijkstra ---------------------------------------------------------------
# Connection length = -log(share of the sending neuron's output). Learning only
# changes KC -> MBON, so the downstream shortest paths from each MBON are computed
# once; then distance(neuron) = min over MBONs of (odor -> MBON) + (MBON -> neuron).
G = nx.DiGraph()


def add_edges(pre_names, post_names, W):              # W[post, pre]
    for p, pre in enumerate(pre_names):
        out = W[:, p]
        for q in np.flatnonzero(out):
            G.add_edge(pre, post_names[q], length=-np.log(out[q] / out.sum()))


mbon_names = [f"M{i}" for i in range(len(m.mbon_ids))]
l1_names = [f"L1_{b}" for b in l1_ids]
l2_names = [f"L2_{b}" for b in l2_ids]
add_edges(mbon_names, l1_names, W1)
add_edges(l1_names, l2_names, W2)
down_names = l1_names + l2_names
D = np.full((len(mbon_names), len(down_names)), np.inf)   # MBON -> downstream distance
col = {n: j for j, n in enumerate(down_names)}
for i, name in enumerate(mbon_names):
    if name in G:
        for n, dist in nx.single_source_dijkstra_path_length(G, name, weight="length").items():
            if n in col:
                D[i, col[n]] = dist


def dijkstra(odor, P, W):
    """Closeness e^(-shortest path) from the odor's active KCs to every downstream neuron."""
    active = np.flatnonzero(m.kc_response(odor, P))
    Wa = W[:, active]
    total = Wa.sum(axis=0)
    Wa, total = Wa[:, total > 0], total[total > 0]
    with np.errstate(divide="ignore"):
        to_mbon = (-np.log(Wa / total)).min(axis=1)        # odor -> each MBON
    return np.exp(-(to_mbon[:, None] + D).min(axis=0))


# ---- Comparing the two -----------------------------------------------------
def spearman(x, y):
    """Rank correlation: +1 = same ordering, 0 = unrelated, -1 = reversed."""
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def one_trial(similarity=0.6, dopamine="punishment", rule="synapse", order="forward",
              details=False):
    """Train odor A, then compare both methods on A, a similar odor B and an unrelated C."""
    P, W = m.pn_to_kc, m.kc_to_mbon.copy()
    A = m.make_odor()
    odors = {"A": A, "B": m.make_odor(A, similarity), "C": m.make_odor()}
    kc_A, kc_B = m.kc_response(A, P), m.kc_response(odors["B"], P)
    methods = {"sim": simulate, "dij": dijkstra}
    before = {(n, k): f(o, P, W) for n, o in odors.items() for k, f in methods.items()}
    m.train(A, P, W, dopamine, rule=rule, order=order)
    after = {(n, k): f(o, P, W) for n, o in odors.items() for k, f in methods.items()}

    row = {"similarity": similarity, "kc_overlap": (kc_A * kc_B).sum() / kc_A.sum()}
    change = {key: after[key] - before[key] for key in before}
    for k in methods:
        dA = change[("A", k)]
        size = dA @ dA
        # size of the memory: how much A's downstream pattern moved, as % of the pattern
        row[f"A_{k}_change"] = np.linalg.norm(dA) / np.linalg.norm(before[("A", k)])
        row[f"transfer_{k}"] = (change[("B", k)] @ dA) / size if size else np.nan
        row[f"C_{k}_spill"] = (change[("C", k)] @ dA) / size if size else np.nan
    row["rank_agreement"] = spearman(after[("A", "sim")], after[("A", "dij")])
    row["change_agreement"] = spearman(change[("A", "sim")], change[("A", "dij")])
    return (row, change) if details else row


if __name__ == "__main__":
    m.rng = np.random.default_rng(0)
    for rule in ("synapse", "threshold"):
        r, change = one_trial(rule=rule, details=True)
        print(f"\n=== punishment, rule={rule}, similarity=0.6, one odor ===")
        print(f"{'':<34}{'simulation':>11}{'Dijkstra':>10}")
        print(f"{'A (trained): size of change':<34}"
              f"{r['A_sim_change']:>11.1%}{r['A_dij_change']:>10.1%}")
        print(f"{'B (similar): transfer':<34}{r['transfer_sim']:>11.0%}{r['transfer_dij']:>10.0%}")
        print(f"{'C (unrelated): spill-over':<34}{r['C_sim_spill']:>11.0%}{r['C_dij_spill']:>10.0%}")
        print(f"Rank agreement (which neurons are most active):      {r['rank_agreement']:.2f}")
        print(f"Change agreement (which neurons the memory changed): {r['change_agreement']:.2f}")
        ds, dd = change[("A", "sim")], change[("A", "dij")]
        dij_rank = np.argsort(np.argsort(-np.abs(dd)))
        print("Neurons the simulation says changed most, and where Dijkstra ranks them:")
        for j in np.argsort(-np.abs(ds))[:5]:
            print(f"  {str(type_of.get(down_ids[j], '?')):<16} ({down_layer[j]}, bodyId {down_ids[j]})"
                  f"  Dijkstra rank {dij_rank[j] + 1} of {len(dd)}")
