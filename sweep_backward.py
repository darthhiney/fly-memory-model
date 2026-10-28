"""
Similarity sweep, 20 trials each: BACKWARD pairing (shock before odor, "relief learning").
Punishment dopamine, synapse rule. Expect odor A to shift mildly toward approach.
Remember: the SIZE of the shift is set by an arbitrary cap in the code; only the
direction is meaningful (see PAPER-NOTES.md).

Needs mb_memory_model.py and sweep_similarity.py in the same folder.
Run:  py sweep_backward.py
"""
from sweep_similarity import run_sweep

run_sweep(dopamine="punishment", rule="synapse", order="backward", name="sweep_backward")
