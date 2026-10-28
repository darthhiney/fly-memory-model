"""
Toy mushroom body memory model (fruit fly).

This runs with made-up wiring that copies the fly's real layout:
    smell neurons (PNs) -> Kenyon cells (KCs) -> output neurons (MBONs) -> approach / avoid
    dopamine neurons (DANs) gate learning in each MBON's "compartment"

Once this makes sense, replace build_random_circuit() with real wiring from neuPrint/FlyWire
(see GUIDE.md).

Run:  python3 mb_memory_model.py
"""
import numpy as np
import networkx as nx

rng = np.random.default_rng(0)

# ---- Circuit sizes (roughly the real fly, one brain side) -----------------
N_PN = 50          # smell channels (glomeruli)
N_KC = 2000        # Kenyon cells
PN_PER_KC = 7      # each KC samples ~7 random smell channels
KC_ACTIVE = 0.05   # APL inhibition keeps only ~5% of KCs active per odor

# Four compartments, each with one MBON and one DAN type.
# Punishment DANs sit on APPROACH MBONs; reward DANs sit on AVOID MBONs.
# Learning weakens KC->MBON synapses where dopamine fires, so after shock the
# odor drives approach less and the fly avoids it.
COMPARTMENTS = [
    # name,   MBON drives,  DAN type
    ("g1",    "approach",  "punishment"),
    ("g2",    "approach",  "punishment"),
    ("b'2",   "avoid",     "reward"),
    ("g5",    "avoid",     "reward"),
]
N_MBON = len(COMPARTMENTS)


def build_random_circuit():
    """PN->KC connections (random) and KC->MBON weights (all start equal)."""
    pn_to_kc = np.zeros((N_KC, N_PN))
    for k in range(N_KC):
        pn_to_kc[k, rng.choice(N_PN, PN_PER_KC, replace=False)] = 1.0
    kc_to_mbon = np.ones((N_MBON, N_KC))
    return pn_to_kc, kc_to_mbon


def make_odor(base=None, similarity=0.0):
    """An odor = firing rate of each smell channel. similarity mixes in a base odor."""
    fresh = rng.random(N_PN) ** 3          # a few channels respond strongly
    return fresh if base is None else similarity * base + (1 - similarity) * fresh


def kc_response(odor, pn_to_kc):
    """Which KCs fire: sum inputs, then keep the top 5% (the APL 'winner-take-most')."""
    drive = pn_to_kc @ odor
    cutoff = np.quantile(drive, 1 - KC_ACTIVE)
    return (drive >= cutoff).astype(float)


def behavior(odor, circuit, gain):
    """Preference index: +1 = full approach, -1 = full avoidance."""
    pn_to_kc, kc_to_mbon = circuit
    mbon = gain * (kc_to_mbon @ kc_response(odor, pn_to_kc))
    approach = sum(mbon[i] for i, c in enumerate(COMPARTMENTS) if c[1] == "approach")
    avoid = sum(mbon[i] for i, c in enumerate(COMPARTMENTS) if c[1] == "avoid")
    return (approach - avoid) / (approach + avoid)


def train(odor, circuit, gain, dopamine, rule="synapse", order="forward",
          trials=5, rate=0.3):
    """
    Pair an odor with punishment or reward dopamine.

    rule="synapse":   weaken KC->MBON synapses of the KCs this odor activates (real fly rule).
    rule="threshold": instead change how easily the whole MBON fires (your threshold idea).
    order="backward": dopamine BEFORE the odor; in real flies this strengthens instead.
    """
    pn_to_kc, kc_to_mbon = circuit
    active = kc_response(odor, pn_to_kc)
    sign = -1 if order == "forward" else +1
    for _ in range(trials):
        for i, (_, _, dan) in enumerate(COMPARTMENTS):
            if dan != dopamine:
                continue
            if rule == "synapse":
                kc_to_mbon[i] *= 1 + sign * rate * active
            else:
                gain[i] *= 1 + sign * rate
    np.clip(kc_to_mbon, 0, 2, out=kc_to_mbon)


def graph_view(odor, circuit, gain):
    """
    Build the circuit as a graph and run Dijkstra.
    Edge length = -log(normalized weight), so the shortest path = the most likely route.
    Returns path length from the odor to 'approach' and to 'avoid'.
    """
    pn_to_kc, kc_to_mbon = circuit
    active = kc_response(odor, pn_to_kc)
    G = nx.DiGraph()
    for k in np.flatnonzero(active):
        G.add_edge("odor", f"KC{k}", length=0.0)
        w = kc_to_mbon[:, k] * gain
        p = w / w.sum() if w.sum() > 0 else w
        for i, (name, drives, _) in enumerate(COMPARTMENTS):
            if p[i] > 0:
                G.add_edge(f"KC{k}", f"MBON-{name}", length=-np.log(p[i]))
                G.add_edge(f"MBON-{name}", drives, length=0.0)
    out = {}
    for target in ("approach", "avoid"):
        # Mean over KCs is more informative than the single best path,
        # since many KCs fire together (see the "many inputs at once" caveat).
        lengths = [nx.dijkstra_path_length(G, f"KC{k}", target, weight="length")
                   for k in np.flatnonzero(active)]
        out[target] = float(np.mean(lengths))
    return out


def experiment(dopamine, rule="synapse", order="forward"):
    circuit = build_random_circuit()
    gain = np.ones(N_MBON)
    odor_A = make_odor()                       # trained odor
    odor_B = make_odor(odor_A, similarity=0.0) # similar odor
    odor_C = make_odor()                       # unrelated odor
    odors = {"A (trained)": odor_A, "B (similar)": odor_B, "C (unrelated)": odor_C}

    before = {n: behavior(o, circuit, gain) for n, o in odors.items()}
    paths_before = graph_view(odor_A, circuit, gain)
    old_w = circuit[1].copy()
    train(odor_A, circuit, gain, dopamine, rule, order)
    after = {n: behavior(o, circuit, gain) for n, o in odors.items()}
    paths_after = graph_view(odor_A, circuit, gain)

    changed = int(np.sum(~np.isclose(old_w, circuit[1])))
    kc_A, kc_B = kc_response(odor_A, circuit[0]), kc_response(odor_B, circuit[0])
    overlap = (kc_A * kc_B).sum() / kc_A.sum()
    print(f"\n=== {dopamine} dopamine, rule={rule}, order={order} ===")
    print(f"{'odor':<15}{'before':>9}{'after':>9}   (+1 approach, -1 avoid)")
    for n in odors:
        print(f"{n:<15}{before[n]:>9.2f}{after[n]:>9.2f}")
    print(f"Synapses changed (the memory trace): {changed} of {old_w.size}")
    print(f"Kenyon cell overlap, B with A: {overlap:.0%} of A's active KCs")
    print("Avg Dijkstra path length from odor A's KCs (shorter = easier route):")
    for t in ("approach", "avoid"):
        print(f"  to {t:<8} before {paths_before[t]:.3f}  after {paths_after[t]:.3f}")


if __name__ == "__main__":
    experiment("punishment")                          # like odor + shock
    experiment("reward")                              # like odor + sugar
    experiment("punishment", order="backward")        # shock BEFORE odor
    experiment("punishment", rule="threshold")        # your threshold idea, for comparison
