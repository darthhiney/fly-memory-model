"""
Similarity sweep, 20 trials each: odor paired with REWARD dopamine (like sugar).
Synapse rule, forward order. Expect odor A to shift toward approach (positive).

Needs mb_memory_model.py and sweep_similarity.py in the same folder.
Run:  py sweep_reward.py
"""
from sweep_similarity import run_sweep

run_sweep(dopamine="reward", rule="synapse", order="forward", name="sweep_reward")
