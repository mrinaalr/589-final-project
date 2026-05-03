import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os

np.random.seed(67)
random.seed(67)


def load(file_path):
    return pd.read_csv(file_path)


def entropy(labels):
    total = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    entropy_value = 0
    for count in counts:
        p = count / total
        entropy_value += -p * np.log2(p)
    return entropy_value


def information_gain(data, attribute_index, label_index, is_numerical=False):
    values    = data.values
    labels    = values[:, label_index]
    attribute = values[:, attribute_index]
    og_entropy = entropy(labels)

    if is_numerical:
        attribute    = attribute.astype(float)
        threshold    = np.mean(attribute)
        left_labels  = labels[attribute <= threshold]
        right_labels = labels[attribute > threshold]
        n            = len(labels)
        new_entropy  = (len(left_labels) / n) * entropy(left_labels) + \
                       (len(right_labels) / n) * entropy(right_labels)
        return og_entropy - new_entropy, threshold

    new_entropy   = 0
    unique_values = np.unique(attribute)
    for val in unique_values:
        rows          = attribute == val
        subset_labels = labels[rows]
        new_entropy  += len(subset_labels) / len(labels) * entropy(subset_labels)
    return og_entropy - new_entropy, None


class Node:
    def __init__(self):
        self.leaf         = False
        self.attribute    = None
        self.children     = {}
        self.label        = None
        self.threshold    = None
        self.is_numerical = False


def build_tree(data, attributes, label_index, numerical_attributes=set()):
    labels = data.values[:, label_index]

    if len(np.unique(labels)) == 1:
        node       = Node()
        node.leaf  = True
        node.label = labels[0]
        return node

    if len(attributes) == 0:
        node       = Node()
        node.leaf  = True
        unique, counts = np.unique(labels, return_counts=True)
        node.label = unique[np.argmax(counts)]
        return node

    best_gain      = 0
    attribute_best = None
    best_threshold = None

    for i in attributes:
        gain, threshold = information_gain(data, i, label_index, i in numerical_attributes)
        if gain > best_gain:
            best_gain      = gain
            attribute_best = i
            best_threshold = threshold

    if attribute_best is None:
        node       = Node()
        node.leaf  = True
        unique, counts = np.unique(labels, return_counts=True)
        node.label = unique[np.argmax(counts)]
        return node

    node              = Node()
    node.attribute    = attribute_best
    node.children     = {}
    unique, counts    = np.unique(labels, return_counts=True)
    node.label        = unique[np.argmax(counts)]
    node.threshold    = best_threshold
    node.is_numerical = attribute_best in numerical_attributes

    values        = data.values
    attribute_col = values[:, attribute_best]
    remaining     = attributes - {attribute_best}

    if node.is_numerical:
        attribute_col = attribute_col.astype(float)
        left_mask     = attribute_col <= best_threshold
        right_mask    = attribute_col >  best_threshold
        if left_mask.any():
            node.children["left"]  = build_tree(data[left_mask],  remaining, label_index, numerical_attributes)
        if right_mask.any():
            node.children["right"] = build_tree(data[right_mask], remaining, label_index, numerical_attributes)
    else:
        for val in np.unique(attribute_col):
            rows               = attribute_col == val
            node.children[val] = build_tree(data[rows], remaining, label_index, numerical_attributes)

    return node


def predict(node, instance):
    if node.leaf:
        return node.label

    attr_val = instance[node.attribute]

    if node.is_numerical:
        attr_val  = float(attr_val)
        child_key = "left" if attr_val <= node.threshold else "right"
        if child_key not in node.children:
            return node.label
        return predict(node.children[child_key], instance)
    else:
        if attr_val not in node.children:
            return node.label
        return predict(node.children[attr_val], instance)


def evaluate(predictions, true_labels):
    """Accuracy + macro-averaged precision, recall, F1 (supports multiclass)."""
    total    = len(true_labels)
    correct  = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / total

    classes    = np.unique(true_labels)
    precisions = []
    recalls    = []
    f1s        = []

    for c in classes:
        TP = sum(1 for p, t in zip(predictions, true_labels) if p == c and t == c)
        FP = sum(1 for p, t in zip(predictions, true_labels) if p == c and t != c)
        FN = sum(1 for p, t in zip(predictions, true_labels) if p != c and t == c)

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return accuracy, float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def bootstrap_sample(data):
    n       = len(data)
    indices = np.random.choice(n, size=n, replace=True)
    values  = data.values
    rows    = [values[i] for i in indices]
    return pd.DataFrame(rows, columns=data.columns)


def build_tree_for_forest(data, attributes, label_index, numerical_attributes=set()):
    labels = data.values[:, label_index]

    if len(np.unique(labels)) == 1:
        node       = Node()
        node.leaf  = True
        node.label = labels[0]
        return node

    if len(attributes) == 0:
        node       = Node()
        node.leaf  = True
        unique, counts = np.unique(labels, return_counts=True)
        node.label = unique[np.argmax(counts)]
        return node

    best_gain      = 0
    attribute_best = None
    best_threshold = None

    m                   = max(1, int(np.sqrt(len(attributes))))
    compared_attributes = set(random.sample(list(attributes), m))

    for i in compared_attributes:
        gain, threshold = information_gain(data, i, label_index, i in numerical_attributes)
        if gain > best_gain:
            best_gain      = gain
            attribute_best = i
            best_threshold = threshold

    if attribute_best is None:
        node       = Node()
        node.leaf  = True
        unique, counts = np.unique(labels, return_counts=True)
        node.label = unique[np.argmax(counts)]
        return node

    node              = Node()
    node.attribute    = attribute_best
    node.children     = {}
    unique, counts    = np.unique(labels, return_counts=True)
    node.label        = unique[np.argmax(counts)]
    node.threshold    = best_threshold
    node.is_numerical = attribute_best in numerical_attributes

    values        = data.values
    attribute_col = values[:, attribute_best]
    remaining     = attributes - {attribute_best}

    if node.is_numerical:
        attribute_col = attribute_col.astype(float)
        left_mask     = attribute_col <= best_threshold
        right_mask    = attribute_col >  best_threshold
        if left_mask.any():
            node.children["left"]  = build_tree_for_forest(data[left_mask],  remaining, label_index, numerical_attributes)
        if right_mask.any():
            node.children["right"] = build_tree_for_forest(data[right_mask], remaining, label_index, numerical_attributes)
    else:
        for val in np.unique(attribute_col):
            rows               = attribute_col == val
            node.children[val] = build_tree_for_forest(data[rows], remaining, label_index, numerical_attributes)

    return node


def predict_majority_forest(forest, instance):
    votes = [predict(tree, instance) for tree in forest]
    unique, counts = np.unique(votes, return_counts=True)
    return unique[np.argmax(counts)]


def stratified(data, index, k):
    labels = np.unique(data.values[:, index])
    folds  = [[] for _ in range(k)]

    for c in labels:
        c_rows  = data[data.iloc[:, index] == c]
        indices = np.random.permutation(len(c_rows))
        shuffled = c_rows.iloc[indices].reset_index(drop=True)
        split_indices = np.array_split(np.arange(len(shuffled)), k)
        for i in range(k):
            folds[i].append(shuffled.iloc[split_indices[i]].reset_index(drop=True))

    for i in range(k):
        folds[i] = pd.concat(folds[i]).reset_index(drop=True)

    return folds


def cross_validate(data, label_index, numerical_attributes, ntree, k=10):
    folds      = stratified(data, label_index, k)
    attributes = set(range(data.shape[1] - 1))

    accuracies = []
    precisions = []
    recalls    = []
    f1s        = []

    for i in range(k):
        test        = folds[i]
        train_parts = [folds[j] for j in range(k) if j != i]
        train       = pd.concat(train_parts).reset_index(drop=True)

        forest = []
        for _ in range(ntree):
            sample = bootstrap_sample(train)
            tree   = build_tree_for_forest(sample, attributes, label_index, numerical_attributes)
            forest.append(tree)

        predictions = [predict_majority_forest(forest, row) for row in test.values]
        true_labels = test.values[:, label_index]

        acc, prec, rec, f1 = evaluate(predictions, true_labels)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return float(np.mean(accuracies)), float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def save_graph(ntree_values, metric_values, dataset_name, metric_name, out_dir):
    plt.figure()
    plt.plot(ntree_values, metric_values, marker='o')
    plt.xlabel("ntree")
    plt.ylabel(metric_name)
    plt.title(f"{dataset_name}: {metric_name} vs ntree")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{dataset_name.lower().replace(' ', '_')}_rf_{metric_name.lower()}.png")
    plt.savefig(fname)
    plt.close()
    print(f"  Saved {fname}")


if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)

    ntree_values = [1, 5, 10, 20, 30, 40, 50]

    datasets = [
        {
            "name":                 "digits",
            "path":                 "preprocessed_datasets/digits_processed.csv",
            "label_index":          64,
            "numerical_attributes": set(range(64)),
        },
        {
            "name":                 "parkinsons",
            "path":                 "preprocessed_datasets/parkinsons_processed.csv",
            "label_index":          22,
            "numerical_attributes": set(range(22)),
        },
        {
            "name":                 "rice",
            "path":                 "preprocessed_datasets/rice_processed.csv",
            "label_index":          7,
            "numerical_attributes": set(range(7)),
        },
        {
            "name":                 "credit",
            "path":                 "preprocessed_datasets/credit_processed.csv",
            "label_index":          46,
            "numerical_attributes": set(range(46)),
        },
        {
            "name":                 "student_dropout",
            "path":                 "preprocessed_datasets/student_dropout_processed.csv",
            "label_index":          254,
            "numerical_attributes": set(range(254)),
        },
    ]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds['name']}")
        print(f"{'='*60}")

        data    = load(ds["path"])
        results = []

        for n in ntree_values:
            print(f"  ntree={n} ... ", end="", flush=True)
            acc, prec, rec, f1 = cross_validate(
                data,
                ds["label_index"],
                ds["numerical_attributes"],
                ntree=n,
                k=10,
            )
            results.append((n, acc, prec, rec, f1))
            print(f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

        print(f"\nRESULTS ({ds['name']}):")
        print(f"{'ntree':>6} | {'accuracy':>9} | {'precision':>9} | {'recall':>7} | {'f1':>7}")
        print("-" * 50)
        for n, acc, prec, rec, f1 in results:
            print(f"{n:>6} | {acc:>9.4f} | {prec:>9.4f} | {rec:>7.4f} | {f1:>7.4f}")

        accs  = [r[1] for r in results]
        precs = [r[2] for r in results]
        recs  = [r[3] for r in results]
        f1s   = [r[4] for r in results]

        save_graph(ntree_values, accs,  ds["name"], "Accuracy",  "graphs")
        save_graph(ntree_values, precs, ds["name"], "Precision", "graphs")
        save_graph(ntree_values, recs,  ds["name"], "Recall",    "graphs")
        save_graph(ntree_values, f1s,   ds["name"], "F1",        "graphs")
