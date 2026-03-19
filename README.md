# Research Project

A structured Python research environment following best practices for reproducibility and modularity.

## Quick Start

1. **Setup environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install numpy pandas matplotlib scikit-learn jupyter ipython python-dotenv
   uv pip install -e .
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start a notebook:**
   - Open `notebooks/00_setup_template.ipynb` as a reference
   - Always use **Cell 0** from the template at the start of any new notebook

## Project Structure

```
├── src/                 Core reusable modules
├── notebooks/           Jupyter notebooks (import from src/)
├── data/
│   ├── tracked/        Small configs & metadata (git-tracked)
│   └── untracked/      Large datasets (git-ignored)
├── results/
│   ├── logs/           Experiment logs
│   └── images/         Plots & visualizations
├── configs/            YAML/TOML experiment configurations
└── other_repos/        External dependencies (git-ignored)
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions and best practices.

## Key Principles

- **No logic in notebooks** → Move to `src/`
- **Config-driven** → Pull hyperparameters from `configs/`
- **Cell 0 setup** → Always initialize notebooks with the standard setup
- **Descriptive naming** → No `temp.py` or `data1.csv`
- **Track smart** → Large files in `data/untracked/`, small metadata in `data/tracked/`

## Documentation

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup & workflows
- [notebooks/00_setup_template.ipynb](notebooks/00_setup_template.ipynb) - Notebook template
- [.env.example](.env.example) - Environment variable template
- [configs/example_experiment.yaml](configs/example_experiment.yaml) - Config template

---

![A_2a_2026121](results/images/stonecat.png)
