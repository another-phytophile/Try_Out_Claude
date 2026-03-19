# Research Project Setup Guide

## Project Structure Overview

```
Research_Repo_Example/
├── src/                    # Core engine - reusable logic
├── notebooks/              # Exploration & execution (logic imported from src/)
├── data/
│   ├── tracked/           # Small metadata, configs, schemas
│   └── untracked/         # Large datasets, caches (git-ignored)
├── results/
│   ├── logs/              # Text logs & telemetry
│   └── images/            # Plots & visualizations
├── configs/               # YAML/TOML hyperparameters & settings
├── other_repos/           # External dependencies
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
└── pyproject.toml         # uv package configuration
```

## Environment Setup

### 1. Create Virtual Environment with uv

```bash
# Initialize project with uv
uv init --name research-project

# Or if you already have pyproject.toml:
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install base dependencies
uv pip install numpy pandas matplotlib scikit-learn jupyter ipython python-dotenv

# Optional: Install ML dependencies
uv pip install torch transformers huggingface-hub

# Install src/ in editable mode
uv pip install -e .
```

### 3. Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env with your actual values
nano .env
```

## Notebook Standards (The "Cell 0" Rule)

**Every notebook MUST start with this Cell 0 setup:**

```python
# %load_ext autoreload
# %autoreload 2

import sys
from pathlib import Path

# Add project root to path for src access
root = Path.cwd().parent
if str(root) not in sys.path:
    sys.path.append(str(root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# IPython Core Magics for efficiency
# %%skip_if_exists (Custom logic for skipping heavy compute chunks)
```

## Coding Principles

### ✅ DO

- **Descriptive Names**: `preprocess_feature_vectors.py`, `baseline_validation_results.log`
- **Move Repeated Logic to src/**: If a function is used more than once, refactor it
- **Pull from configs/**: Use YAML/TOML files for hyperparameters, not hardcoded values
- **Keep data/untracked/ clean**: Never commit large datasets or caches

### ❌ DON'T

- Define logic in notebooks that could be reused
- Hardcode paths or hyperparameters
- Commit files to `data/untracked/` or `results/`
- Use temporary names like `temp.py` or `data1.csv`

## Module Structure in src/

Create organized, descriptive modules:

```
src/
├── __init__.py
├── data_processing.py          # Data loading & preprocessing
├── feature_engineering.py       # Feature extraction & transformation
├── models.py                    # Model definitions
├── evaluation.py                # Evaluation metrics & reporting
├── utils/
│   ├── __init__.py
│   ├── config.py               # Config loading
│   └── logging.py              # Custom logging
└── visualization.py             # Plotting utilities
```

## Example Workflow

### 1. Create a Data Processing Module

**src/data_processing.py:**
```python
import pandas as pd
from pathlib import Path

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean features."""
    return df.fillna(0).astype(float)
```

### 2. Use in Notebook

**notebooks/01_data_exploration.ipynb:**
```python
# Cell 0: Setup
# %load_ext autoreload
# %autoreload 2

import sys
from pathlib import Path
root = Path.cwd().parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from dotenv import load_dotenv
load_dotenv()

# Cell 1: Import & Execute
from src.data_processing import load_dataset, preprocess_features
import os

data_path = os.getenv("DATA_DIR")
df = load_dataset(f"{data_path}/raw_data.csv")
df_processed = preprocess_features(df)
```

### 3. Pull Hyperparameters from Config

**configs/experiment.yaml:**
```yaml
seed: 42
batch_size: 32
learning_rate: 0.001
epochs: 10
```

**src/utils/config.py:**
```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

## Git Workflow

### Don't Commit Large Files
```bash
git status
# Check data/untracked/ and results/ are NOT showing

git add src/ notebooks/ configs/ .env.example
git commit -m "Add feature engineering module"
git push
```

### Track Progress
- Use `results/logs/` for experiment logs (git-ignored but locally visible)
- Keep `data/tracked/` for small metadata only
- Store processed data references in `configs/`

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:** Add Cell 0 setup to your notebook:
```python
import sys
from pathlib import Path
root = Path.cwd().parent
sys.path.append(str(root))
```

### Issue: Changes in src/ not reflecting in notebook

**Solution:** Use autoreload magic in Cell 0:
```python
%load_ext autoreload
%autoreload 2
```

### Issue: `.env` variables not loading

**Solution:** Call in Cell 0:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Quick Reference: File Destinations

| Content | Location | Track? | Notes |
|---------|----------|--------|-------|
| Python modules | `src/` | ✅ | Always import from here |
| Notebooks | `notebooks/` | ✅ | Use Cell 0 setup |
| Hyperparameters | `configs/` | ✅ | YAML/TOML files |
| Raw datasets | `data/untracked/` | ❌ | Large files |
| Metadata/schemas | `data/tracked/` | ✅ | Small JSON/YAML |
| Experiment logs | `results/logs/` | ❌ | Text-based telemetry |
| Plots/charts | `results/images/` | ❌ | PNG/PDF exports |
| External repos | `other_repos/` | ❌ | Git clones |

---

**Happy researching! 🚀**
