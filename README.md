# 641-music-recsys

Term project for DSCI 641 (Recommender Systems) at Drexel University, exploring collaborative filtering on music listening data with LensKit.

**Authors:** Phillip Roman, Ryan Quinlan

## Project Scope

Scope TBD. We are currently evaluating candidate datasets and refining research questions. This section will be updated once the project direction is finalized.

## Software Setup

This project uses:

- [`uv`][uv] for managing Python environments and dependencies
- Python 3.12 (managed by uv)
- [LensKit 2026](https://lkpy.readthedocs.io/) for recommender algorithms and evaluation
- [DVC](https://dvc.org/) for data and artifact versioning
- Visual Studio Code (recommended but not required)

[uv]: https://astral.sh/uv/

### Installation

1. Install `uv`:

   - **Mac:** `brew install uv`
   - **Windows:** `winget install astral-sh.uv`
   - **Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. Clone the repository and sync dependencies:

   ```console
   $ git clone https://github.com/PhillipJRoman/641-music-recsys.git
   $ cd 641-music-recsys
   $ uv sync
   ```

   `uv sync` creates a virtual environment in `.venv` with all project dependencies.

3. Activate the environment:

   ```console
   $ source .venv/bin/activate
   ```

   On Windows: `.venv\Scripts\activate`

### Pulling Data

Data and intermediate artifacts are versioned with DVC.

> **TBD:** DVC remote storage and pull instructions will be added once the data pipeline is configured.

## Directory Layout

```
641-music-recsys/
  data/            input datasets (tracked with DVC, not git)
  src/             Python package with shared code (imported by scripts and notebooks)
  scripts/         stage-based scripts (split, train, evaluate, etc.)
  notebooks/       analysis notebooks, load artifacts and produce results
  results/         generated outputs (rec lists, metrics, plots)
  pyproject.toml   project configuration and dependencies
  dvc.yaml         DVC pipeline definition
  README.md        this file
```

## Workflow

The project follows a script-per-stage architecture:

1. Splitting, training, and batch inference run as standalone scripts, each producing artifacts saved to disk.
2. Notebooks load the saved artifacts and produce analysis, metrics, and plots.
3. DVC tracks artifacts and enables both authors to share results without re-running expensive computations.

## Contact

Phillip Roman — pjr322@drexel.edu

## License

Released under the MIT License. See `LICENSE` for details.
