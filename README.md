# Shortest-path memory maps in the fly mushroom body

Code, results and figures for the paper **"Shortest-path memory maps in the fly mushroom body"** by Ayan Mehra.

The project asks a simple question. If you look up a memory in a brain wiring diagram with a shortest-path algorithm (Dijkstra), how accurate is the "memory map" you get? It tests this in the fruit fly mushroom body in two ways:

- a **toy model** with random wiring, and
- a **real model** built from the [hemibrain connectome](https://neuprint.janelia.org) (v1.2.1).

Each memory map is checked against a simulation of activity in the same circuit, and real wiring is compared with 100 shuffled wirings.

The repository already contains every result CSV used in the paper (`Data/`) and the final figures (`Figures/`). You can redraw the figures and recompute every statistic in about a minute, with no internet and no account.

---

## Quick start: redraw the figures (no download needed)

You need [Python 3](https://www.python.org/downloads/) (3.10 or newer).

1. Download this repository. Click the green **Code** button, choose **Download ZIP**, and unzip it. If you use git, run `git clone https://github.com/darthhiney/fly-memory-model` instead.
2. Open a terminal in the unzipped folder.
   - **Windows:** open the folder in File Explorer, type `cmd` in the address bar, and press Enter.
   - **Mac/Linux:** open Terminal and `cd` into the folder.
3. Install the libraries (one time only):
   ```
   py -m pip install -r requirements.txt        (Windows)
   python3 -m pip install -r requirements.txt   (Mac/Linux)
   ```
4. Draw the figures and compute the statistics:
   ```
   py make_figures.py          (Windows)
   python3 make_figures.py     (Mac/Linux)
   ```

The script writes `fig1`–`fig5` and `statistics.txt` into `Figures/`. These should match the copies already in the repository.

## Re-running the experiments from scratch

Every script uses fixed random seeds, so a full re-run should reproduce the CSVs in `Data/`. Each script saves its CSV files in the folder you run it from. To redraw the figures from your new results, move those files into `Data/` (any subfolder works) or delete the old ones there. `make_figures.py` finds each file by name and uses the newest copy.

### Toy model (no download needed)

| Script | What it runs | Paper section |
| --- | --- | --- |
| `mb_memory_model.py` | One toy circuit learning one odor (a demo) | Method |
| `sweep_similarity.py` | Generalization curve: punishment, synapse rule | Results 1, Fig. 1 |
| `sweep_threshold.py` | Same, with the excitability ("threshold") rule | Results 1, Fig. 1 |
| `sweep_reward.py` | Same, with reward dopamine | Method |
| `sweep_backward.py` | Same, with dopamine *before* the odor | Method |
| `sweep_dijkstra_vs_sim.py` | Dijkstra map vs. simulation, 0% and 30% inhibition | Results 3, Fig. 4 |
| `sweep_inhibition.py` | How Dijkstra fails as inhibition goes from 0% to 50% | Results 3, Table 2 |

`dijkstra_vs_sim.py` holds the shared code for the last two sweeps.

### Real hemibrain model (needs a free neuPrint account)

The wiring data is not included here. You download it yourself from neuPrint:

1. Make a free account at [neuprint.janelia.org](https://neuprint.janelia.org) and copy your API token from your account page.
2. Put the token in an environment variable. **Never paste it into a file you share or upload.**
   ```
   set NEUPRINT_APPLICATION_CREDENTIALS=your_token_here       (Windows)
   export NEUPRINT_APPLICATION_CREDENTIALS=your_token_here    (Mac/Linux)
   ```
3. Download the wiring (one time only):
   ```
   py fetch_hemibrain.py      -> neurons.csv, pn_kc.csv, kc_mbon.csv
   py fetch_downstream.py     -> mbon_l1.csv, l1_l2.csv, down_neurons.csv
   ```
4. Run the experiments:

| Script | What it runs | Paper section |
| --- | --- | --- |
| `mb_real_model.py` | One learning run on real wiring (top MBONs, baseline) | Results 2 |
| `sweep_real.py` | Generalization curves, real vs. one shuffled wiring | Results 2 |
| `sweep_real_vs_shuffles.py` | Real wiring vs. 100 shuffled wirings (takes several minutes) | Results 2, Fig. 2 |
| `sweep_dijkstra_vs_sim_real.py` | Dijkstra map vs. simulation on real downstream neurons | Results 3, Fig. 5 |

`dijkstra_vs_sim_real.py` holds the shared code for the last sweep.

## What's in the folders

- `Data/`: every result CSV used in the paper. `*_raw.csv` holds one row per trial, and `*_summary.csv` holds the means for each similarity level.
  - `Smell Specificity/`: generalization curves for the toy and real models.
  - `Real vs Shuffled/`: real wiring vs. 100 shuffled circuits.
  - `Dijkstra vs Simulation/`: the shortest-path map vs. the activity simulation.
  - `Validation and Baseline/`: printed output of `mb_real_model.py`.
- `Figures/`: the paper's figures and `statistics.txt`, which lists every reported number with its confidence interval and p-value.
- `Archive/`: an earlier 20-shuffle pilot, replaced by the 100-shuffle run. It is kept outside `Data/` so `make_figures.py` doesn't pick it up.

Figure files and their figure numbers in the paper:

| File | Paper figure |
| --- | --- |
| `fig1_threshold_vs_synapse.png` | Figure 1 |
| `fig2_real_vs_shuffled.png` | Figure 2 |
| `fig5_generalization.png` | Figure 3 |
| `fig3_dijkstra_toy.png` | Figure 4 |
| `fig4_dijkstra_real.png` | Figure 5 |

## Data source and credit

Wiring data comes from the hemibrain connectome: Scheffer, L. K., et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife, 9*, e57443. https://doi.org/10.7554/eLife.57443. It was accessed through [neuPrint](https://neuprint.janelia.org) with [neuprint-python](https://github.com/connectome-neuprint/neuprint-python).

## Contact

Ayan Mehra, ayankmehra@gmail.com
