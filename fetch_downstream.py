"""
Download the REAL neurons downstream of the mushroom body output neurons (MBONs),
two steps deep, from neuPrint (hemibrain), for dijkstra_vs_sim_real.py.

    MBONs -> L1 (every neuron an MBON connects to) -> L2 (every neuron L1 connects to)

Run once, in the same folder as neurons.csv (from fetch_hemibrain.py):
    set NEUPRINT_APPLICATION_CREDENTIALS=your_token_here
    py fetch_downstream.py

It saves:
    mbon_l1.csv        MBON -> L1 connections (synapse counts)
    l1_l2.csv          L1 -> L2 connections (synapse counts)
    down_neurons.csv   bodyId and type of every L1 and L2 neuron, so results have names
"""
import pandas as pd
from neuprint import Client, fetch_adjacencies

client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1")

# Only keep connections with at least this many synapses. 10 is a common cutoff for
# "strong" hemibrain connections; lower = more neurons and a slower model.
MIN_WEIGHT = 10

neurons = pd.read_csv("neurons.csv")
mb = neurons[neurons["class"].isin(["KC", "MBON"])]["bodyId"]
mbon_ids = neurons[(neurons["class"] == "MBON") & neurons["compartment"].notna()]["bodyId"].tolist()


def downstream(sources, label):
    print(f"Downloading {label} connections (this can take a few minutes)...")
    found, conn = fetch_adjacencies(sources, None, min_total_weight=MIN_WEIGHT)
    # neuPrint splits each connection by brain region; add the pieces back together
    conn = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    conn = conn[conn["weight"] >= MIN_WEIGHT]
    return found[["bodyId", "type"]], conn


types1, mbon_l1 = downstream(mbon_ids, "MBON -> L1")
mbon_l1 = mbon_l1[~mbon_l1["bodyId_post"].isin(mb)]          # L1 = not a KC or MBON
l1_ids = mbon_l1["bodyId_post"].unique().tolist()
print(f"  {len(l1_ids)} L1 neurons")

types2, l1_l2 = downstream(l1_ids, "L1 -> L2")
l1_l2 = l1_l2[~l1_l2["bodyId_post"].isin(mb) & ~l1_l2["bodyId_post"].isin(l1_ids)]
print(f"  {l1_l2['bodyId_post'].nunique()} L2 neurons")

mbon_l1.to_csv("mbon_l1.csv", index=False)
l1_l2.to_csv("l1_l2.csv", index=False)
pd.concat([types1, types2]).drop_duplicates("bodyId").to_csv("down_neurons.csv", index=False)
print("Done. Saved mbon_l1.csv, l1_l2.csv, down_neurons.csv")
