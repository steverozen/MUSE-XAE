# MUSE-XAE — Working Notes for Claude Code

This is Steve Rozen's fork of [MUSE-XAE](https://doi.org/10.1093/bioinformatics/btae320) (mutational-signature extraction with an explainable autoencoder; SBS-96 only). The upstream code is unchanged; the only local additions are environment/build infrastructure to get it running reproducibly on Steve's laptop and on the **Duke HPC cluster (DCC)**.

Read this file before doing any work in the repo.

## Goal of the local fork

Make MUSE-XAE installable and runnable from one lockfile on:

1. Steve's Linux laptop (Zorin/GNOME, no GPU) — for smoke tests.
2. **Duke HPC (DCC), GPU node** — for real de-novo runs on tumor catalogs.

Upstream's install instructions (`conda create … python=3.10 numpy=1.24.3 && pip install -r requirements.txt`) are brittle: `requirements.txt` pins `lap==0.4.0`, which no longer builds on modern `setuptools` (missing `pkg_resources`). We replaced that flow with **pixi**.

## Environment: pixi (not conda+pip)

- `pixi.toml` + `pixi.lock` are the source of truth. Same lockfile drives laptop and HPC.
- `[dependencies]` currently pins:
  - `python = 3.10.*`
  - `pip >= 26`
  - `lap >= 0.5.13` (from conda-forge — replaces the broken `lap==0.4.0` pin). `models.py` imports `from lap import lapjv`; that path is unchanged in 0.5.x, so no source edit needed.
- The rest of the deps from `requirements.txt` (tensorflow, scikit-learn, optuna, pandas, numpy 1.24.x, etc.) are **not yet added** to `pixi.toml`. Next step is to migrate them in (prefer conda-forge; fall back to `[pypi-dependencies]` only when a wheel is missing). Do this incrementally and re-run `pixi install` after each addition so a single bad pin is easy to bisect.
- Channel: `conda-forge` only.
- Platform: `linux-64` only (no macOS/Windows targets).

Local setup:
```
cd ~/github/MUSE-XAE
pixi install
pixi shell           # or: pixi run python ./MUSE-XAE/main.py …
```

## HPC migration — status & next steps

**Status: not yet started on DCC.** Steve has been working locally; this commit (`8b96311`) is the handoff point.

When you (Claude on the HPC) pick this up:

1. **Clone the repo** on DCC and `pixi install`. Pixi is self-bootstrapping — install via `curl -fsSL https://pixi.sh/install.sh | bash` if it's not already on the cluster.
2. **GPU**: upstream README says "code runs only on CPUs." `requirements.txt` listed `tensorflow-cpu==2.11.0` plus a full set of `nvidia-*-cu11` wheels — those CUDA wheels are vestigial. For HPC GPU runs, swap `tensorflow-cpu` for the GPU-capable `tensorflow` build matched to the DCC CUDA module. **Verify the TF↔CUDA↔cuDNN matrix against whatever module DCC currently provides** before adding to `pixi.toml`. TF 2.11 expects CUDA 11.2 / cuDNN 8.1; if DCC only offers newer CUDA, bump TF accordingly and test that `models.py` still imports cleanly (it uses `tf.keras.optimizers.legacy.Adam` and `disable_eager_execution`, both of which can break across major TF versions).
3. **Slurm**: no batch script exists yet. Write one under `scripts/` (create the dir) — request a GPU node, load the CUDA module, `pixi run python ./MUSE-XAE/main.py --dataset <X> …`. The `--n_jobs` arg defaults to 24 (de-novo) / 12 (refit); match to the allocated `--cpus-per-task`.
4. **Outputs** land in `Experiments/` (gitignored). On HPC, point `--directory` at scratch / a project space, not `$HOME`.

## Repo layout (don't get confused by the doubled name)

```
MUSE-XAE/                  ← repo root (this dir)
├── MUSE-XAE/              ← actual Python package; main.py lives here
│   ├── main.py            ← CLI entry point
│   ├── models.py          ← autoencoder definition; lap.lapjv used here
│   ├── de_novo.py         ← de-novo extraction loop
│   ├── assignment.py      ← refitting
│   └── …
├── datasets/              ← input catalogs (96 × N CSV/TSV)
├── Experiments/           ← run outputs (gitignored)
├── notebook/              ← reproducibility notebooks from upstream
├── pixi.toml / pixi.lock  ← env (local addition)
├── requirements.txt       ← legacy upstream pins; do NOT install from this
└── README.md              ← upstream README; install section is stale
```

## Conventions

- Don't edit `requirements.txt` — it's kept for reference but is no longer the install path. README still references it; that's a known stale section.
- Don't commit anything under `Experiments/` or `.pixi/`.
- Co-author tag on commits made via Claude:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Steve's git identity: `Steven G. Rozen <steverozen@pm.me>` (note: pm.me, not gmail).

## Open questions / decisions deferred

- Whether to keep `tensorflow-cpu` as a fallback dep alongside the GPU TF on HPC, or split into two pixi `environments` / `features`. Defer until first GPU run works.
- Whether to upstream the pixi setup (PR to original repo) or keep as a local-only fork. Steve has not decided.
- COSMIC reference version: upstream defaults to v3.4. No reason to change yet.

## Related context outside this repo

This work is part of Steve's broader **NeuralSig / mutsig grant** effort. See `~/MEGA/ea/projects/mutsig_grant_2026/` and the memory note `project_neuralsig_hpc_migration.md` in `~/.claude/projects/-home-steve-MEGA-ea/memory/` for the wider picture (R01 VAE code is a separate scaffold; MUSE-XAE here is the established baseline being benchmarked against).
