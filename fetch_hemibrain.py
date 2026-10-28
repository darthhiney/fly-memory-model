"""
Stage 1, part 1: download the real mushroom body wiring from neuPrint (hemibrain).

Run once (needs internet and your neuPrint token):
    py -m pip install neuprint-python pandas
    set NEUPRINT_APPLICATION_CREDENTIALS=your_token_here      (Windows; Mac: export ...)
    py fetch_hemibrain.py

It saves three files next to this script, which mb_real_model.py then uses:
    neurons.csv   every PN, KC, MBON and dopamine neuron, with its type and compartment
    pn_kc.csv     smell neuron -> Kenyon cell connections (synapse counts)
    kc_mbon.csv   Kenyon cell -> output neuron connections (synapse counts)
"""
import re
import pandas as pd
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

# If you didn't set the environment variable, paste your token here instead: token="..."
client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1")

PN = NC(type=".*PN.*", regex=True)        # smell projection neurons, e.g. DA1_lPN
KC = NC(type="KC.*", regex=True)          # Kenyon cells
MBON = NC(type="MBON.*", regex=True)      # mushroom body output neurons
DAN = NC(type="(PAM|PPL1).*", regex=True) # dopamine: PAM ~ reward, PPL1 ~ punishment

# Mushroom body compartments are ROIs named like g1(R), a'2(R), b'2(R)
COMPARTMENT = re.compile(r"^[abg]'?\d\(R\)$")


def main_compartment(roi_counts, column):
    """For each neuron, the compartment where it has the most synapses."""
    rc = roi_counts[roi_counts["roi"].str.match(COMPARTMENT)]
    best = rc.sort_values(column, ascending=False).drop_duplicates("bodyId")
    return best.set_index("bodyId")["roi"]


def connections(source, target, label):
    print(f"Downloading {label} connections...")
    _, conn = fetch_adjacencies(source, target)
    # neuPrint splits each connection by brain region; add the pieces back together
    return conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()


print("Downloading neuron lists...")
tables = []
for crit, cls in [(PN, "PN"), (KC, "KC"), (MBON, "MBON"), (DAN, "DAN")]:
    neurons, rois = fetch_neurons(crit)
    df = neurons[["bodyId", "type"]].copy()
    df["class"] = cls
    if cls == "MBON":   # MBONs receive KC input in their compartment
        df["compartment"] = df["bodyId"].map(main_compartment(rois, "post"))
    elif cls == "DAN":  # DANs send output into their compartment
        df["compartment"] = df["bodyId"].map(main_compartment(rois, "pre"))
    tables.append(df)
pd.concat(tables).to_csv("neurons.csv", index=False)

connections(PN, KC, "PN -> KC").to_csv("pn_kc.csv", index=False)
connections(KC, MBON, "KC -> MBON").to_csv("kc_mbon.csv", index=False)
print("Done. Saved neurons.csv, pn_kc.csv, kc_mbon.csv")
