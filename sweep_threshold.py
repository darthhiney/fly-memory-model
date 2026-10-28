"""
Similarity sweep, 20 trials each: THRESHOLD rule (your original idea) instead of synapses.
Punishment dopamine, forward order. The whole output neuron becomes harder to activate,
for every input. Prediction: transfer is ~100% at every similarity (a flat curve),
because the change isn't tied to odor A's Kenyon cells.

Needs mb_memory_model.py and sweep_similarity.py in the same folder.
Run:  py sweep_threshold.py
"""
from sweep_similarity import run_sweep

run_sweep(dopamine="punishment", rule="threshold", order="forward", name="sweep_threshold")
