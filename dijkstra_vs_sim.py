"""
Dijkstra vs simulation: how well does a cheap shortest-path "memory map" match a
simulation of activity flowing through the same circuit?

The toy mushroom body (mb_memory_model.py) is extended with two downstream layers,
so paths are several steps long and Dijkstra has real work to do:

    odor -> KCs -> 4 MBONs -> L1 (40 neurons) -> L2 (16 neurons) -> APPROACH / AVOID

Half of L1/L2 sit on the approach side and half on the avoid side. A fraction of L1
neurons are INHIBITORY and suppress the opposite side, like real MBON-to-MBON and
downstream inhibition.

Two ways to read the same circuit:
  SIMULATION  each neuron adds up its inputs (inhibitory ones count negative) and
              fires above a small threshold. Activity flows layer by layer.
  DIJKSTRA    each connection's length = -log(share of the sending neuron's output).
              closeness(neuron) = e^(-shortest path length from the odor's KCs).
              Dijkstra can't use negative weights, so inhibitory connections are
              treated as if they were excitatory. That's a known weakness we measure.

Run:  py dijkstra_vs_sim.py
Needs mb_memory_model.py in the same folder.
"""
import numpy as np
import networkx as nx
import mb_memory_model as m

N_L1, N_L2 = 40, 16
L1_INPUTS = 2          # MBONs feeding each L1 neuron
L2_INPUTS = 6          # L1 neurons feeding each L2 neuron
INHIBITORY = 0.3       # fraction of L1 neurons that are inhibitory (0 = none)
FIRE_THRESHOLD = 0.05  # simulation: activity below this (relative) doesn't pass on

SIDE = {"approach": [i for i, c in enumerate(m.COMPARTMENTS) if c[1] == "approach"],
        "avoid": [i for i, c in enumerate(m.COMPARTMENTS) if c[1] == "avoid"]}


def build_downstream(rng):
    """Random but valence-respecting wiring below the MBONs."""
    l1_side = np.array(["approach"] * (N_L1 // 2) + ["avoid"] * (N_L1 // 2))
    l1_inhib = rng.random(N_L1) < INHIBITORY
    W1 = np.zeros((N_L1, m.N_MBON))            # MBON -> L1
    for j in range(N_L1):
        W1[j, rng.choice(SIDE[l1_side[j]], L1_INPUTS)] += rng.uniform(0.5, 1.5, L1_INPUTS)
    l2_side = np.array(["approach"] * (N_L2 // 2) + ["avoid"] * (N_L2 // 2))
    W2 = np.zeros((N_L2, N_L1))                # L1 -> L2
    for j in range(N_L2):
        same = np.flatnonzero((l1_side == l2_side[j]) & ~l1_inhib)
        opp_inhib = np.flatnonzero((l1_side != l2_side[j]) & l1_inhib)
        pool = np.concatenate([same, opp_inhib])
        pick = rng.choice(pool, min(L2_INPUTS, len(pool)), replace=False)
        W2[j, pick] = rng.uniform(0.5, 1.5, len(pick))
    W3 = np.array([[1.0 if s == side else 0.0 for s in l2_side]
                   for side in ("approach", "avoid")])  # L2 -> outputs
    return {"W1": W1, "W2": W2, "W3": W3, "l1_inhib": l1_inhib}


def simulate(odor, circuit, gain, down):
    """Activity flowing layer by layer. Returns rates of every downstream neuron."""
    kc = m.kc_response(odor, circuit[0])
    mbon = gain * (circuit[1] @ kc)
    mbon = mbon / mbon.max()                      # relative units
    l1 = np.maximum(down["W1"] @ mbon - FIRE_THRESHOLD, 0)
    sign = np.where(down["l1_inhib"], -1.0, 1.0)
    l2 = np.maximum(down["W2"] @ (sign * l1) - FIRE_THRESHOLD, 0)
    out = down["W3"] @ l2
    return {"L1": l1, "L2": l2, "approach": out[0], "avoid": out[1]}


def dijkstra(odor, circuit, gain, down):
    """Closeness e^(-shortest path) from the odor's active KCs to every downstream neuron."""
    pn_to_kc, kc_to_mbon = circuit
    active = np.flatnonzero(m.kc_response(odor, pn_to_kc))
    G = nx.DiGraph()

    def add_edges(pre_names, post_names, W):           # W[post, pre]
        for p, pre in enumerate(pre_names):
            out = np.abs(W[:, p])
            if out.sum() == 0:
                continue
            for q in np.flatnonzero(out):
                G.add_edge(pre, post_names[q], length=-np.log(out[q] / out.sum()))

    mbon_names = [f"MBON{i}" for i in range(m.N_MBON)]
    l1_names = [f"L1_{j}" for j in range(N_L1)]
    l2_names = [f"L2_{j}" for j in range(N_L2)]
    for k in active:
        G.add_edge("odor", f"KC{k}", length=0.0)
    add_edges([f"KC{k}" for k in active], mbon_names, (kc_to_mbon * gain[:, None])[:, active])
    add_edges(mbon_names, l1_names, down["W1"])
    add_edges(l1_names, l2_names, down["W2"])
    add_edges(l2_names, ["approach", "avoid"], down["W3"])
    dist = nx.single_source_dijkstra_path_length(G, "odor", weight="length")
    close = lambda n: np.exp(-dist[n]) if n in dist else 0.0
    return {"L1": np.array([close(n) for n in l1_names]),
            "L2": np.array([close(n) for n in l2_names]),
            "approach": close("approach"), "avoid": close("avoid")}


def preference(r):
    total = r["approach"] + r["avoid"]
    return 0.0 if total == 0 else (r["approach"] - r["avoid"]) / total


def spearman(x, y):
    """Rank correlation: +1 = same ordering, 0 = unrelated, -1 = reversed."""
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def one_trial(similarity=0.6, dopamine="punishment", rule="synapse", order="forward"):
    """Train odor A, then compare both methods on A, a similar odor B and an unrelated C."""
    circuit = m.build_random_circuit()
    gain = np.ones(m.N_MBON)
    down = build_downstream(m.rng)
    A = m.make_odor()
    odors = {"A": A, "B": m.make_odor(A, similarity), "C": m.make_odor()}
    before = {n: (simulate(o, circuit, gain, down), dijkstra(o, circuit, gain, down))
              for n, o in odors.items()}
    m.train(A, circuit, gain, dopamine, rule=rule, order=order)
    after = {n: (simulate(o, circuit, gain, down), dijkstra(o, circuit, gain, down))
             for n, o in odors.items()}

    row = {"similarity": similarity}
    for n in odors:
        (s0, d0), (s1, d1) = before[n], after[n]
        row[f"{n}_sim_shift"] = preference(s1) - preference(s0)
        row[f"{n}_dij_shift"] = preference(d1) - preference(d0)
    for method in ("sim", "dij"):
        a = row[f"A_{method}_shift"]
        row[f"transfer_{method}"] = row[f"B_{method}_shift"] / a if a else np.nan
    # Neuron-level agreement on odor A, after learning
    s1, d1 = after["A"]
    row["rank_agreement"] = spearman(np.concatenate([s1["L1"], s1["L2"]]),
                                     np.concatenate([d1["L1"], d1["L2"]]))
    # Does Dijkstra predict WHICH neurons the memory changed?
    s0, d0 = before["A"]
    ds = np.concatenate([s1["L1"] - s0["L1"], s1["L2"] - s0["L2"]])
    dd = np.concatenate([d1["L1"] - d0["L1"], d1["L2"] - d0["L2"]])
    row["change_agreement"] = spearman(ds, dd)
    return row


if __name__ == "__main__":
    m.rng = np.random.default_rng(0)
    for rule in ("synapse", "threshold"):
        r = one_trial(rule=rule)
        print(f"\n=== punishment, rule={rule}, similarity=0.6, inhibitory={INHIBITORY:.0%} ===")
        print(f"{'odor':<15}{'sim shift':>11}{'Dijkstra shift':>16}")
        for n, label in (("A", "A (trained)"), ("B", "B (similar)"), ("C", "C (unrelated)")):
            print(f"{label:<15}{r[f'{n}_sim_shift']:>11.2f}{r[f'{n}_dij_shift']:>16.2f}")
        print(f"Transfer to B:   simulation {r['transfer_sim']:.0%}   Dijkstra {r['transfer_dij']:.0%}")
        print(f"Rank agreement (which neurons are most active):  {r['rank_agreement']:.2f}")
        print(f"Change agreement (which neurons the memory changed): {r['change_agreement']:.2f}")
