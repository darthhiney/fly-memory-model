"""
Step 5: draw the paper's figures and compute the statistics, from every CSV you've made.

Run from your main project folder (the one with the Data folder):
    py -m pip install matplotlib      (once)
    py make_figures.py

It finds each CSV by its file name anywhere inside Data (or the main folder), so your
folder names don't matter. A figure whose CSVs are missing is skipped with a message.
Output, in a new Figures folder:
    fig1_threshold_vs_synapse.png   claim 1: the trained odor can't tell the rules apart
    fig2_real_vs_shuffled.png       claim 2: real wiring vs many shuffled wirings
    fig3_dijkstra_toy.png           claim 5: Dijkstra vs simulation, toy circuit
    fig4_dijkstra_real.png          claim 5: Dijkstra vs simulation, real wiring
    fig5_generalization.png         generalization curves vs similarity AND KC overlap
    statistics.txt                  every number and test the figures rest on
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "Figures"
N_PERM = 10000                       # random sign flips per permutation test
rng = np.random.default_rng(0)

# Colors: blue = simulation / real wiring / synapse rule; orange = Dijkstra / threshold rule
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9a94", "#2b2b29"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.8, "lines.linewidth": 2, "lines.markersize": 5,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

os.makedirs(OUT, exist_ok=True)
report = []


def say(text=""):
    print(text)
    report.append(text)


def find(name):
    """Load a CSV by file name from anywhere under Data (or the current folder)."""
    hits = glob.glob(os.path.join("Data", "**", name), recursive=True) + glob.glob(name)
    # if the same file name exists twice (an older run), use the newest one
    return pd.read_csv(max(hits, key=os.path.getmtime)) if hits else None


def need(*names):
    tables = [find(n) for n in names]
    missing = [n for n, t in zip(names, tables) if t is None]
    if missing:
        say(f"  (skipped: couldn't find {', '.join(missing)})")
        return None
    return tables


def paired_test(diff):
    """Mean of paired differences, 95% bootstrap interval, and a sign-flip permutation p-value."""
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    mean = diff.mean()
    boots = rng.choice(diff, (N_PERM, len(diff))).mean(axis=1)
    flips = rng.choice([-1.0, 1.0], (N_PERM, len(diff)))
    p = (np.sum(np.abs((flips * diff).mean(axis=1)) >= abs(mean)) + 1) / (N_PERM + 1)
    return mean, np.percentile(boots, [2.5, 97.5]), p, len(diff)


def report_test(label, diff):
    mean, (lo, hi), p, n = paired_test(diff)
    ptext = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    say(f"  {label}: mean difference {mean:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), "
        f"{ptext}, n = {n} trials")


def curve(ax, x, y, err, color, label, marker="o"):
    ax.errorbar(x, y, yerr=err, color=color, label=label, marker=marker, capsize=0,
                elinewidth=1, alpha=1)


# ---- Figure 1 / claim 1 -------------------------------------------------------------
say("CLAIM 1: memory maps must be checked on untrained odors")
t = need("similarity_sweep_summary.csv", "sweep_threshold_summary.csv",
         "dvs_synapse_inhib0_raw.csv", "dvs_threshold_inhib0_raw.csv")
if t:
    syn, thr, dsyn, dthr = t
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    curve(a, syn.kc_overlap_mean, syn.transfer_mean, syn.transfer_spread, BLUE, "synapse rule")
    curve(a, thr.kc_overlap_mean, thr.transfer_mean, thr.transfer_spread, ORANGE, "threshold rule", "s")
    a.set(xlabel="Kenyon cell overlap of B with A", ylabel="Transfer to similar odor B",
          title="a  Only the synapse rule is odor-specific")
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.legend()
    groups = ["trained odor A", "unrelated odor C"]
    vals = {"synapse rule": [dsyn.A_dij_shift.mean(), dsyn.C_dij_shift.mean()],
            "threshold rule": [dthr.A_dij_shift.mean(), dthr.C_dij_shift.mean()]}
    errs = {"synapse rule": [dsyn.A_dij_shift.std(), dsyn.C_dij_shift.std()],
            "threshold rule": [dthr.A_dij_shift.std(), dthr.C_dij_shift.std()]}
    x = np.arange(2)
    for i, (k, c) in enumerate([("synapse rule", BLUE), ("threshold rule", ORANGE)]):
        b.bar(x + (i - 0.5) * 0.36, vals[k], 0.34, yerr=errs[k], color=c, label=k,
              error_kw={"elinewidth": 1, "ecolor": INK})
    b.axhline(0, color=GRAY, lw=1)
    b.set_xticks(x, groups)
    b.set(ylabel="Preference shift in the Dijkstra map", title="b  Identical for A, different for C")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(os.path.join(OUT, "fig1_threshold_vs_synapse.png"))
    plt.close(fig)
    n = min(len(dsyn), len(dthr))
    same = np.abs(dsyn.A_dij_shift[:n].values - dthr.A_dij_shift[:n].values)
    say(f"  Trained odor A, Dijkstra map: largest difference between rules over {n} paired "
        f"trials = {same.max():.2e} (identical)")
    report_test("Unrelated odor C, threshold minus synapse (Dijkstra)",
                dthr.C_dij_shift[:n].values - dsyn.C_dij_shift[:n].values)

# ---- Figure 2 / claim 2 -------------------------------------------------------------
say("\nCLAIM 2: real wiring vs shuffled wirings")
t = need("shuffles_punishment_curves.csv", "shuffles_reward_curves.csv")
if t:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.6]})
    for ax, (name, c) in zip(axes[:2], zip(["punishment", "reward"], t)):
        shuf = c[c.circuit != "real"].pivot(index="similarity", columns="circuit", values="transfer")
        real = c[c.circuit == "real"].set_index("similarity").transfer
        ax.fill_between(shuf.index, shuf.min(axis=1), shuf.max(axis=1), color=GRAY, alpha=0.35,
                        lw=0, label=f"range of {shuf.shape[1]} shuffles")
        ax.plot(real.index, real.values, color=BLUE, marker="o", label="real wiring")
        ax.set(xlabel="Odor similarity", ylabel="Transfer to similar odor B",
               title=f"{'a' if name == 'punishment' else 'b'}  {name.capitalize()} learning")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.legend()
        inside = (real >= shuf.min(axis=1)) & (real <= shuf.max(axis=1))
        test = real.index < 1.0
        say(f"  {name}: real transfer outside the shuffled range at {int((~inside[test]).sum())} "
            f"of {int(test.sum())} similarity levels (chance expects about "
            f"{test.sum() * 2 / (shuf.shape[1] + 1):.1f})")
    c = t[0]
    base = c.groupby("circuit").baseline.mean()
    real_b, shuf_b = base["real"], base.drop("real").values
    ax = axes[2]
    ax.scatter(np.zeros(len(shuf_b)) + rng.uniform(-0.08, 0.08, len(shuf_b)), shuf_b, color=GRAY,
               s=30, label="shuffled")
    ax.scatter([0], [real_b], color=BLUE, s=70, zorder=3, label="real")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set(ylabel="Untrained preference (random odors)", title="c  Innate bias")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    fig.savefig(os.path.join(OUT, "fig2_real_vs_shuffled.png"))
    plt.close(fig)
    lower = np.sum(shuf_b <= real_b)
    say(f"  Untrained baseline: real {real_b:.3f}, shuffled {shuf_b.mean():.3f} "
        f"(range {shuf_b.min():.3f} to {shuf_b.max():.3f}); {lower} of {len(shuf_b)} shuffles at or "
        f"below real, one-sided p = {(lower + 1) / (len(shuf_b) + 1):.3f} "
        f"(the smallest p possible with {len(shuf_b)} shuffles)")

# ---- Figure 3 / claim 5, toy ----------------------------------------------------------
say("\nCLAIM 5 (toy circuit): Dijkstra vs simulation")
t = need("dvs_synapse_inhib0_summary.csv", "dvs_synapse_inhib0_raw.csv", "inhibition_sweep_summary.csv")
if t:
    s, r, inh = t
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    curve(a, s.similarity, s.transfer_sim_mean, s.transfer_sim_spread, BLUE, "simulation")
    curve(a, s.similarity, s.transfer_dij_mean, s.transfer_dij_spread, ORANGE, "Dijkstra", "s")
    a.set(xlabel="Odor similarity", ylabel="Transfer to similar odor B",
          title="a  Dijkstra's transfer is flat")
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.legend()
    curve(b, s.similarity, s.C_sim_shift_mean, s.C_sim_shift_spread, BLUE, "simulation")
    curve(b, s.similarity, s.C_dij_shift_mean, s.C_dij_shift_spread, ORANGE, "Dijkstra", "s")
    b.axhline(0, color=GRAY, lw=1)
    b.set(xlabel="Odor similarity (of B; C is always unrelated)", ylabel="Preference shift, odor C",
          title="b  Dijkstra invents a memory for C")
    b.legend()
    c.plot(inh.inhibitory, inh.captured, color=ORANGE, marker="s")
    c.axhline(1, color=GRAY, lw=1, ls="--")
    c.set(xlabel="Inhibitory neurons in layer 1", ylabel="Memory captured by Dijkstra",
          title="c  Inhibition hides the memory")
    c.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    c.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.savefig(os.path.join(OUT, "fig3_dijkstra_toy.png"))
    plt.close(fig)
    report_test("No inhibition, odor C shift, Dijkstra minus simulation", r.C_dij_shift - r.C_sim_shift)
    report_test("No inhibition, odor A shift, Dijkstra minus simulation", r.A_dij_shift - r.A_sim_shift)
    for lo, hi, label in ((0, 0.4, "low similarity (0-0.4)"), (0.85, 0.95, "high similarity (0.85-0.95)")):
        sel = r[(r.similarity >= lo) & (r.similarity <= hi)]
        report_test(f"Transfer at {label}, Dijkstra minus simulation", sel.transfer_dij - sel.transfer_sim)
    say("  Share of memory captured by inhibition level: " + ", ".join(
        f"{lv:.0%} -> {cp:.0%}" for lv, cp in zip(inh.inhibitory, inh.captured)))

# ---- Figure 4 / claim 5, real wiring --------------------------------------------------
say("\nCLAIM 5 (real wiring): Dijkstra vs simulation")
t = need("dvsreal_punishment_summary.csv", "dvsreal_reward_summary.csv",
         "dvsreal_punishment_raw.csv", "dvsreal_reward_raw.csv")
if t:
    sp, sr, rp, rr = t
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, s, name in ((a, sp, "a  Punishment"), (b, sr, "b  Reward")):
        curve(ax, s.kc_overlap_mean, s.transfer_sim_mean, s.transfer_sim_spread, BLUE, "simulation")
        curve(ax, s.kc_overlap_mean, s.transfer_dij_mean, s.transfer_dij_spread, ORANGE, "Dijkstra", "s")
        ax.set(xlabel="Kenyon cell overlap of B with A", ylabel="Transfer to similar odor B", title=name)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.legend()
    x = np.arange(2)
    for i, (col, lab, colr) in enumerate((("C_sim_spill", "simulation", BLUE), ("C_dij_spill", "Dijkstra", ORANGE))):
        c.bar(x + (i - 0.5) * 0.36, [rp[col].mean(), rr[col].mean()], 0.34,
              yerr=[rp[col].std(), rr[col].std()], color=colr, label=lab,
              error_kw={"elinewidth": 1, "ecolor": INK})
    c.set_xticks(x, ["punishment", "reward"])
    c.set(ylabel="Spill-over to unrelated odor C", title="c  Spill-over to unrelated odors")
    c.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(os.path.join(OUT, "fig4_dijkstra_real.png"))
    plt.close(fig)
    for name, r in (("punishment", rp), ("reward", rr)):
        report_test(f"{name}: odor C spill-over, Dijkstra minus simulation", r.C_dij_spill - r.C_sim_spill)
        sel = r[(r.similarity > 0) & (r.similarity < 1)]
        report_test(f"{name}: transfer, Dijkstra minus simulation", sel.transfer_dij - sel.transfer_sim)
        say(f"  {name}: rank agreement {r.rank_agreement.mean():.2f}, "
            f"change agreement {r.change_agreement.mean():.2f}")

# ---- Figure 5 / generalization curves -------------------------------------------------
say("\nGENERALIZATION CURVES and BASELINE (claims 2 and 4)")
t = need("similarity_sweep_summary.csv", "real_punishment_real_summary.csv",
         "real_reward_real_summary.csv", "real_punishment_real_raw.csv")
if t:
    toy, rp, rw, raw = t
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    series = ((toy, "toy, punishment", GRAY, "^"), (rp, "real, punishment", BLUE, "o"),
              (rw, "real, reward", ORANGE, "s"))
    for s, lab, colr, mk in series:
        curve(a, s.similarity, s.transfer_mean, s.transfer_spread, colr, lab, mk)
        curve(b, s.kc_overlap_mean, s.transfer_mean, s.transfer_spread, colr, lab, mk)
    a.set(xlabel="Odor similarity (mixing fraction)", ylabel="Transfer to similar odor B",
          title="a  Against similarity")
    b.set(xlabel="Kenyon cell overlap of B with A", ylabel="Transfer to similar odor B",
          title="b  Against Kenyon cell overlap")
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    for ax in (a, b):
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.legend()
    fig.savefig(os.path.join(OUT, "fig5_generalization.png"))
    plt.close(fig)
    say(f"  Untrained baseline, real wiring: {raw.A_before.mean():.3f} +/- {raw.A_before.std():.3f} "
        f"over {len(raw)} random odors")

with open(os.path.join(OUT, "statistics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
print(f"\nSaved figures and statistics.txt in the {OUT} folder")
