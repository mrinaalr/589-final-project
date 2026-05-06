# 589 Final Project

Classification experiments on the preprocessed CSVs in `preprocessed_datasets/`.

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Run commands from this directory (repo root).

## Run

**Random forest** — 10-fold CV across tree counts; metrics to the terminal, plots under `graphs/`:

```bash
python rf.py
```

**Ensemble** (random forest + neural net) — CV and summary bars in `graphs/`:

```bash
python ensemble.py
```

**Notebooks** — open `knn.ipynb` or `preprocess.ipynb` or `adaptive_knn.ipynb` in Jupyter and run the cells.
