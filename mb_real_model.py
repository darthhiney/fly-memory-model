"""
Stage 1, part 2: the same memory experiment as mb_memory_model.py, on REAL wiring.

Run fetch_hemibrain.py first, then:
    py mb_real_model.py

What's real here: every PN->KC and KC->MBON connection and its synapse count, and
which dopamine type (PAM = reward, PPL1 = punishment) covers each compartment.
What's still simplified: odors are random patterns over smell channels, the APL
inhibition is the "top 5% of KCs" shortcut, and neurons just add up their inputs.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
KC_ACTIVE = 0.05

neurons = pd.read_csv("neurons.csv")
pn_kc = pd.read_csv("pn_kc.csv")
kc_mbon = pd.read_csv("kc_mbon.csv")

# ---- Turn the tables into matrices -----------------------------------------
pns = neurons[neurons["class"] == "PN"]
kcs = neurons[neurons["class"] == "KC"]["bodyId"].to_numpy()
mbons = neurons[(neurons["class"] == "MBON") & neurons["compartment"].notna()]
kc_index = {b: i for i, b in enumerate(kcs)}

# Group PNs by smell channel (glomerulus = the part of the type before "_")
pns = pns.assign(glom=pns["type"].str.split("_").str[0])
gloms = sorted(pns["glom"].unique())
glom_of_pn = dict(zip(pns["bodyId"], pns["glom"]))
glom_index = {g: i for i, g in enumerate(gloms)}

pn_to_kc = np.zeros((len(kcs), len(gloms)))
for pre, post, w in pn_kc[["bodyId_pre", "bodyId_post", "weight"]].itertuples(index=False):
    if post in kc_index and pre in glom_of_pn:
        pn_to_kc[kc_index[post], glom_index[glom_of_pn[pre]]] += w

mbon_ids = mbons["bodyId"].to_numpy()
mbon_index = {b: i for i, b in enumerate(mbon_ids)}
kc_to_mbon = np.zeros((len(mbon_ids), len(kcs)))
for pre, post, w in kc_mbon[["bodyId_pre", "bodyId_post", "weight"]].itertuples(index=False):
    if pre in kc_index and post in mbon_index:
        kc_to_mbon[mbon_index[post], kc_index[pre]] += w

# ---- Which dopamine covers each compartment, and so what each MBON drives ---
dans = neurons[(neurons["class"] == "DAN") & neurons["compartment"].notna()]
dan_kind = {}
for comp, group in dans.groupby("compartment"):
    pam = group["type"].str.startswith("PAM").sum()
    ppl1 = group["type"].str.startswith("PPL1").sum()
    dan_kind[comp] = "reward" if pam >= ppl1 else "punishment"

mbon_dan = np.array([dan_kind.get(c, "none") for c in mbons["compartment"]])
# Learning weakens an MBON's input, so an MBON in a punishment compartment must
# normally drive approach (weakening it -> avoidance), and vice versa.
mbon_drives = np.where(mbon_dan == "punishment", "approach",
                       np.where(mbon_dan == "reward", "avoid", "none"))

print(f"Loaded {len(gloms)} smell channels, {len(kcs)} KCs, {len(mbon_ids)} MBONs")
print("Compartments and their dopamine:", dict(sorted(dan_kind.items())))


# ---- Same experiment as the toy model ---------------------------------------
N_ODORS = 20        # repeat with this many random odors and average, to separate effects from luck
SIMILARITY = 0.6    # how much of odor A is mixed into odor B
SHUFFLE_CONTROL = True  # also run on randomly shuffled wiring for comparison


def make_odor(base=None, similarity=0.0):
    fresh = rng.random(len(gloms)) ** 3
    return fresh if base is None else similarity * base + (1 - similarity) * fresh


def kc_response(odor, P):
    drive = P @ odor
    return (drive >= np.quantile(drive, 1 - KC_ACTIVE)).astype(float)


def sides(odor, P, W):
    out = W @ kc_response(odor, P)
    return out[mbon_drives == "approach"].sum(), out[mbon_drives == "avoid"].sum()


# In the real wiring, avoid-side MBONs get more KC synapses in total than the
# approach side, so untrained odors score negative. That's the honest output of
# "just add up synapses". BALANCE = True is an ASSUMPTION, not data: it scales the
# avoid side so an average random odor scores 0, as if something downstream that
# the connectome doesn't show evens the two sides out.
BALANCE = False


def avoid_scale(P, W):
    if not BALANCE:
        return 1.0
    test = np.array([sides(make_odor(), P, W) for _ in range(100)])
    return test[:, 0].mean() / test[:, 1].mean()


def behavior(odor, P, W, scale):
    approach, avoid = sides(odor, P, W)
    avoid *= scale
    return (approach - avoid) / (approach + avoid)


def train(odor, P, W, dopamine, trials=5, rate=0.3, rule="synapse", order="forward"):
    """
    rule="synapse":   weaken KC->MBON synapses of the KCs this odor activates (real fly rule).
    rule="threshold": change how easily the whole MBON fires, for every input.
    order="backward": dopamine BEFORE the odor, which strengthens instead (capped at 2x start).
    """
    active = kc_response(odor, P)
    rows = mbon_dan == dopamine
    sign = -1 if order == "forward" else +1
    start = W.copy()
    for _ in range(trials):
        if rule == "synapse":
            W[rows] *= 1 + sign * rate * active
        else:
            W[rows] *= 1 + sign * rate
    np.minimum(W, 2 * start, out=W)


def shuffled(P, W):
    """
    Random wiring with the same ingredients: each KC keeps its own input strengths
    but they come from random smell channels, and each MBON keeps its own synapse
    counts but they come from random KCs. Anything the real wiring does that this
    doesn't is evidence that the fly's specific wiring matters.
    """
    P2 = np.array([rng.permutation(row) for row in P])
    W2 = np.array([rng.permutation(row) for row in W])
    return P2, W2


def experiment(dopamine, P, W0, label):
    names = ["A (trained)", "B (similar)", "C (unrelated)"]
    before, after, changed, drops = [], [], [], np.zeros(len(W0))
    scale = avoid_scale(P, W0)
    for _ in range(N_ODORS):
        W = W0.copy()
        A = make_odor()
        odors = [A, make_odor(A, SIMILARITY), make_odor()]
        before.append([behavior(o, P, W, scale) for o in odors])
        train(A, P, W, dopamine)
        after.append([behavior(o, P, W, scale) for o in odors])
        changed.append(np.sum(~np.isclose(W0, W)))
        drops += (W0 - W).sum(axis=1)
    before, after = np.array(before), np.array(after)
    shift = after - before
    print(f"\n=== {dopamine} dopamine, {label}, averaged over {N_ODORS} odors ===")
    print(f"{'odor':<15}{'before':>8}{'after':>8}{'shift':>8}  (+/- spread)")
    for i, n in enumerate(names):
        print(f"{n:<15}{before[:, i].mean():>8.2f}{after[:, i].mean():>8.2f}"
              f"{shift[:, i].mean():>8.2f}  +/- {shift[:, i].std():.2f}")
    gen = shift[:, 1].mean() / shift[:, 0].mean()
    print(f"Similar odor moved {gen:.0%} as far as the trained odor")
    print(f"Connections changed per memory: {np.mean(changed):.0f} of {int(np.sum(W0 > 0))}")
    top = np.argsort(drops)[::-1][:5]
    print("Biggest changes:", ", ".join(
        f"{mbons['type'].iloc[i]} ({mbons['compartment'].iloc[i]})" for i in top if drops[i] > 0))


if __name__ == "__main__":
    for dop in ("punishment", "reward"):
        experiment(dop, pn_to_kc, kc_to_mbon, "REAL wiring")
        if SHUFFLE_CONTROL:
            experiment(dop, *shuffled(pn_to_kc, kc_to_mbon), "SHUFFLED wiring")
