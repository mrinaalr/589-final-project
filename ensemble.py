import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os

from rf import (build_tree_for_forest, predict_majority_forest, stratified, evaluate)
from sklearn.preprocessing import StandardScaler

np.random.seed(67)
random.seed(67)


# ── NeuralNet (inlined from nn.py — avoids its Windows-path top-level code) ──

def one_hot(y, num_classes):
    y = y.astype(int)
    mat = np.zeros((num_classes, y.shape[0]))
    mat[y, np.arange(y.shape[0])] = 1
    return mat


class NeuralNet:
    def __init__(self, layers, reg_lambda=0.0):
        self.L         = len(layers) - 1
        self.sizes     = layers
        self.reg_lambda = reg_lambda
        self.W = [np.random.uniform(-1, 1, (layers[i+1], layers[i]))
                  for i in range(self.L)]
        self.b = [np.random.uniform(-1, 1, (layers[i+1], 1))
                  for i in range(self.L)]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_deriv(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def forward(self, x):
        A, activations, Zs = x, [x], []
        for i in range(self.L):
            Z = self.W[i] @ A + self.b[i]
            A = self._sigmoid(Z)
            Zs.append(Z)
            activations.append(A)
        return activations, Zs

    def _backward(self, activations, Zs, y):
        n     = y.shape[1]
        dW    = [None] * self.L
        db    = [None] * self.L
        deltas = [None] * self.L
        deltas[-1] = activations[-1] - y
        dW[-1] = (deltas[-1] @ activations[-2].T) / n + (self.reg_lambda / n) * self.W[-1]
        db[-1] = np.sum(deltas[-1], axis=1, keepdims=True) / n
        for l in reversed(range(self.L - 1)):
            deltas[l] = (self.W[l+1].T @ deltas[l+1]) * self._sigmoid_deriv(Zs[l])
            dW[l] = (deltas[l] @ activations[l].T) / n + (self.reg_lambda / n) * self.W[l]
            db[l] = np.sum(deltas[l], axis=1, keepdims=True) / n
        return dW, db

    def train(self, X, Y, epochs=500, lr=0.1, batch_size=32):
        m = X.shape[1]
        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X, Y = X[:, perm], Y[:, perm]
            for i in range(0, m, batch_size):
                Xb, Yb = X[:, i:i+batch_size], Y[:, i:i+batch_size]
                acts, Zs = self.forward(Xb)
                dW, db   = self._backward(acts, Zs, Yb)
                for j in range(self.L):
                    self.W[j] -= lr * dW[j]
                    self.b[j] -= lr * db[j]

    def predict(self, X, threshold=0.5):
        activations, _ = self.forward(X)
        output = activations[-1]
        if output.shape[0] == 1:
            return (output >= threshold).astype(int).flatten()
        return np.argmax(output, axis=0)


# ── KNN (ported from knn.ipynb) ───────────────────────────────────────────────

def _knn_single(X_train, Y_train, test_point, k):
    dists   = np.sqrt(np.sum((X_train.values - test_point.values) ** 2, axis=1))
    indices = np.argsort(dists)[:k]
    return Y_train.iloc[indices].value_counts().index[0]


def knn_predict_all(X_train, Y_train, X_test, k):
    return np.array([_knn_single(X_train, Y_train, X_test.iloc[i], k)
                     for i in range(len(X_test))], dtype=int)


# ── NN train/predict helpers ──────────────────────────────────────────────────

NN_HIDDEN = [3, 3]
NN_REG    = 0.01
NN_EPOCHS = 500


def train_nn(X_df, Y_series, n_classes):
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_df)
    layers = NN_HIDDEN.copy()
    layers.insert(0, X_df.shape[1])
    layers.append(1 if n_classes == 2 else n_classes)
    net  = NeuralNet(layers, reg_lambda=NN_REG)
    Y_np = Y_series.to_numpy().astype(int)
    if n_classes == 2:
        net.train(X_norm.T, Y_np.reshape(1, -1).astype(float), epochs=NN_EPOCHS)
    else:
        net.train(X_norm.T, one_hot(Y_np, n_classes), epochs=NN_EPOCHS)
    return net, scaler


def predict_nn(net, scaler, X_df):
    return net.predict(scaler.transform(X_df).T).astype(int)


# ── Bootstrap + majority vote ─────────────────────────────────────────────────

def bootstrap(df):
    idx = np.random.choice(len(df), size=len(df), replace=True)
    return df.iloc[idx].reset_index(drop=True)


def majority_vote(a, b, c):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    c = np.asarray(c, dtype=int)
    result = np.empty(len(a), dtype=int)
    for i in range(len(a)):
        votes = [a[i], b[i], c[i]]
        unique, counts = np.unique(votes, return_counts=True)
        result[i] = unique[np.argmax(counts)]
    return result


# ── Ensemble cross-validation ─────────────────────────────────────────────────

def ensemble_cross_validate(data, label_index, numerical_attributes,
                             k=10, knn_k=5, ntree=10):
    n_classes  = len(np.unique(data.iloc[:, label_index]))
    folds      = stratified(data, label_index, k)
    attributes = set(range(data.shape[1] - 1))
    label_col  = data.columns[label_index]
    feat_cols  = [c for c in data.columns if c != label_col]

    accuracies, precisions, recalls, f1s = [], [], [], []

    for fold_i in range(k):
        test  = folds[fold_i]
        train = pd.concat([folds[j] for j in range(k) if j != fold_i]).reset_index(drop=True)

        # Separate bootstrap sample per algorithm
        rf_boot  = bootstrap(train)
        knn_boot = bootstrap(train)
        nn_boot  = bootstrap(train)

        # ── RF ───────────────────────────────────────────────────────────────
        forest = [build_tree_for_forest(rf_boot, attributes, label_index, numerical_attributes)
                  for _ in range(ntree)]

        # ── KNN (min-max normalised on its own bootstrap) ─────────────────────
        X_knn_tr = knn_boot[feat_cols]
        Y_knn_tr = knn_boot[label_col]
        lo, hi   = X_knn_tr.min(), X_knn_tr.max()
        denom    = (hi - lo).replace(0, 1)
        X_knn_tr_norm = ((X_knn_tr - lo) / denom).fillna(0)

        # ── NN (StandardScaler fitted on its own bootstrap) ───────────────────
        net, scaler = train_nn(nn_boot[feat_cols], nn_boot[label_col], n_classes)

        # ── Predict on held-out fold ─────────────────────────────────────────
        X_test = test[feat_cols]
        Y_test = test[label_col].to_numpy().astype(int)

        pred_rf  = np.array([predict_majority_forest(forest, row) for row in test.values],
                             dtype=int)

        X_test_norm = ((X_test - lo) / denom).fillna(0)
        pred_knn = knn_predict_all(X_knn_tr_norm, Y_knn_tr, X_test_norm, knn_k)

        pred_nn  = predict_nn(net, scaler, X_test)

        preds = majority_vote(pred_rf, pred_knn, pred_nn)

        acc, prec, rec, f1 = evaluate(preds, Y_test)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f"  fold {fold_i+1}/{k} — acc={acc:.4f}  f1={f1:.4f}")

    return (float(np.mean(accuracies)), float(np.mean(precisions)),
            float(np.mean(recalls)),    float(np.mean(f1s)))


# ── Graph ─────────────────────────────────────────────────────────────────────

def save_bar(dataset_name, acc, prec, rec, f1, out_dir):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values  = [acc, prec, rec, f1]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(metrics, values, color=['steelblue', 'salmon', 'seagreen', 'goldenrod'])
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Ensemble: {dataset_name}")
    ax.set_ylabel("Score")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.4f}", ha='center', fontsize=9)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{dataset_name.lower().replace(' ', '_')}_ensemble.png")
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")


# ── Dataset configs ───────────────────────────────────────────────────────────

DATASETS = [
    {
        "name":                 "digits",
        "path":                 "preprocessed_datasets/digits_processed.csv",
        "label_rename":         None,
        "label_index":          64,
        "numerical_attributes": set(range(64)),
    },
    {
        "name":                 "parkinsons",
        "path":                 "preprocessed_datasets/parkinsons_processed.csv",
        "label_rename":         "Diagnosis",
        "label_index":          22,
        "numerical_attributes": set(range(22)),
    },
    {
        "name":                 "rice",
        "path":                 "preprocessed_datasets/rice_processed.csv",
        "label_rename":         None,
        "label_index":          7,
        "numerical_attributes": set(range(7)),
    },
    {
        "name":                 "credit",
        "path":                 "preprocessed_datasets/credit_processed.csv",
        "label_rename":         None,
        "label_index":          46,
        "numerical_attributes": set(range(46)),
    },
    {
        "name":                 "student_dropout",
        "path":                 "preprocessed_datasets/student_dropout_processed.csv",
        "label_rename":         "Target",
        "label_index":          254,
        "numerical_attributes": set(range(254)),
    },
]


if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)
    all_results = []

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds['name']}")
        print(f"{'='*60}")

        data = pd.read_csv(ds["path"])
        if ds["label_rename"]:
            data = data.rename(columns={ds["label_rename"]: "label"})
        data["label"] = data["label"].astype(int)

        acc, prec, rec, f1 = ensemble_cross_validate(
            data,
            ds["label_index"],
            ds["numerical_attributes"],
        )
        all_results.append((ds["name"], acc, prec, rec, f1))
        save_bar(ds["name"], acc, prec, rec, f1, "graphs")

    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'dataset':<20} | {'accuracy':>9} | {'precision':>9} | {'recall':>7} | {'f1':>7}")
    print("-" * 60)
    for name, acc, prec, rec, f1 in all_results:
        print(f"{name:<20} | {acc:>9.4f} | {prec:>9.4f} | {rec:>7.4f} | {f1:>7.4f}")
