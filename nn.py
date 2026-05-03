import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def one_hot(y, num_classes):
    y = y.astype(int)
    one_hot = np.zeros((num_classes, y.shape[0]))
    one_hot[y, np.arange(y.shape[0])] = 1
    return one_hot
def split(data): ##one hot encodes + normalizes + splits data
    X = data.drop(columns=['label'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test
def best_harmonic_index(scores): ##for selecting best model
    h_scores = [
        2 * acc * f1 / (acc + f1) if (acc + f1) != 0 else 0
        for acc, f1 in scores
    ]
    return int(np.argmax(h_scores))
class NeuralNet:
    def __init__(self, layers, reg_lambda=0.0): ##initialize
        self.L = len(layers) - 1 
        self.sizes = layers
        self.reg_lambda = reg_lambda

        self.W = []
        self.b = []
        self.t = []

        for i in range(self.L):
            w = np.random.uniform(-1, 1,(layers[i+1], layers[i]))
            b = np.random.uniform(-1, 1,(layers[i+1], 1))
            self.W.append(w)
            self.b.append(b)

    def backprop_test(self,w,b): ## for unit testing
        self.W=w
        self.B=b

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, x):
        A = x
        activations = [A]
        Zs = []

        for i in range(self.L):
            Z = self.W[i] @ A + self.b[i]
            A = self.sigmoid(Z)

            Zs.append(Z)
            activations.append(A)

        return activations, Zs

    def compute_loss(self, y_hat, y):
        n = y.shape[1]
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9) 

        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / n
        reg = (self.reg_lambda / (2 * n)) * sum(np.sum(W**2) for W in self.W)

        return loss + reg

    def backward(self, activations, Zs, y):
        n = y.shape[1]

        dW = [None] * self.L
        db = [None] * self.L
        deltas = [None] * self.L
        deltas[-1] = activations[-1] - y

        dW[-1] = (deltas[-1] @ activations[-2].T) / n + (self.reg_lambda / n) * self.W[-1]
        db[-1] = np.sum(deltas[-1], axis=1, keepdims=True) / n

        for l in reversed(range(self.L - 1)):
            deltas[l] = (self.W[l+1].T @ deltas[l+1]) * self.sigmoid_deriv(Zs[l])
            dW[l] = (deltas[l] @ activations[l].T) / n + (self.reg_lambda / n) * self.W[l]
            db[l] = np.sum(deltas[l], axis=1, keepdims=True) / n

        return dW, db
 
    def update(self, dW, db, lr):
        for i in range(self.L):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def train(self, X, Y, epochs=1000, lr=0.1, batch_size=32):
        m = X.shape[1]
        for epoch in range(epochs):
            perm = np.random.permutation(m)
            X, Y = X[:, perm], Y[:, perm]

            for i in range(0, m, batch_size):
                X_batch = X[:, i:i+batch_size]
                Y_batch = Y[:, i:i+batch_size]
                activations, Zs = self.forward(X_batch)
                dW, db = self.backward(activations, Zs, Y_batch)
                self.update(dW, db, lr)

            if epoch % 100 == 0:
                activations, _ = self.forward(X)
                loss = self.compute_loss(activations[-1], Y)
                self.t.append((epoch,loss))
                #print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X, threshold=0.5):
        activations, _ = self.forward(X)
        output = activations[-1]

        if output.shape[0] == 1:
            return (output >= threshold).astype(int).flatten()
        else:
            return np.argmax(output, axis=0)

    def plot(self, title="Training Loss", xlabel="Epoch", ylabel="Loss",config=None,test_acc=None,test_f1=None):
        
        epochs, losses = zip(*self.t)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        caption_parts = []
        if config:
            caption_parts.append(f"Config: {config}")
        if test_acc is not None:
            caption_parts.append(f"Test Acc: {test_acc:.4f}")
        if test_f1 is not None:
            caption_parts.append(f"Test F1: {test_f1:.4f}")
        caption = " | ".join(caption_parts)
        if caption:
            plt.figtext(
                0.5, -0.05, caption,
                ha="center",
                fontsize=9,
                wrap=True
            )
        plt.tight_layout()
        plt.show()

def stratified_k_fold(Y, k=10):
    folds=[[] for _ in range(k)]
    
    for label in Y.unique():
        indices=np.where(Y==label)[0]
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i%k].append(idx)
    
    return folds 

configs = [[[3,3],0.01,2000],
           [[5,5],0.01,2000],
           [[7,7],0.01,2000],
           [[3,7],0.01,2000],
           [[7,3],0.01,2000],
           [[3,5,7],0.01,2000],
           ]

def evaluate_net(X, Y, n_folds=10):
    np.random.seed(67)  
    folds=stratified_k_fold(Y, n_folds)
    ret_vals = []
    for config in configs:
      accuracies=[]
      f1_scores=[]
      for i in range(n_folds):
        test_idx=folds[i]
        train_idx =[idx for j in range(n_folds) if j!=i for idx in folds[j]]
        
        X_train, X_test=X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test=Y.iloc[train_idx], Y.iloc[test_idx]
        classes = len(np.unique(Y_train))
        #Normalization
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm  = scaler.transform(X_test)
        
        layers = config[0].copy()
        layers.insert(0, X_train.shape[1])
        if classes == 2:
            layers.append(1)
        else:
            layers.append(classes)

        net = NeuralNet(layers,reg_lambda=config[1])
        if classes != 2:
            Y_train = one_hot(Y_train,classes)
            net.train(X_train_norm.T, Y_train, epochs=config[2])
        else:
            net.train(X_train_norm.T, Y_train.to_numpy().reshape(1, -1), epochs=config[2])

        
        #Getting the predictions
        X_test_np = X_test_norm.T          # shape (n_features, m)
        Y_test_np = Y_test.to_numpy().reshape(1, -1)  # shape (1, m)
        preds = net.predict(X_test_np)            # shape (1, m)
        
        #Compute accuracy and F1
        acc = np.mean(preds == Y_test_np.flatten())
        if classes == 2:
          f1 = f1_score(Y_test_np.flatten(), preds)  # binary
        else:
          f1 = f1_score(Y_test_np.flatten(), preds, average='macro')  # multiclass

        accuracies.append(acc)
        f1_scores.append(f1)
      ret_vals.append((np.mean(accuracies), np.mean(f1_scores)))
    return ret_vals

def best_model(data):
    X_train,Y_train,X_test,Y_test = split(data)
    best_idx = best_harmonic_index(evaluate_net(X_train,Y_train))
    net,scaler = make_model(X_train,Y_train,configs[best_idx])
    acc,f1 = scores(scaler,X_test,Y_test,net)
    print(acc,f1,configs[best_idx])
    net.plot(config=configs[best_idx],test_acc=acc,test_f1=f1)
    return net

def make_model(X_train,Y_train,config):
  classes = len(np.unique(Y_train))
  scaler = StandardScaler()
  X_train_norm = scaler.fit_transform(X_train)   
  layers = config[0].copy()
  layers.insert(0, X_train.shape[1])
  if classes == 2:
      layers.append(1)
  else:
      layers.append(classes)

  net = NeuralNet(layers,reg_lambda=config[1])
  if classes != 2:
      Y_train = one_hot(Y_train,classes)
      net.train(X_train_norm.T, Y_train, epochs=config[2])
  else:
      net.train(X_train_norm.T, Y_train.to_numpy().reshape(1, -1), epochs=config[2])
  return net,scaler

def scores(scaler,X_test,Y_test,net):
  #Getting the predictions

  X_test_np = scaler.transform(X_test).T          # shape (n_features, m)
  Y_test_np = Y_test.to_numpy().reshape(1, -1)  # shape (1, m)
  preds = net.predict(X_test_np)            # shape (1, m)
  
  #Compute accuracy and F1
  acc = np.mean(preds == Y_test_np.flatten())
  if net.sizes[-1] == 1:
    f1 = f1_score(Y_test_np.flatten(), preds) # binary
  else:
    f1 = f1_score(Y_test_np.flatten(), preds, average='macro')  # multiclass
  return acc,f1

credit_data = pd.read_csv(os.path.join(os.getcwd(), "pre_processed_datasets\\credit_processed.csv"))
digit_data = pd.read_csv(os.path.join(os.getcwd(), "pre_processed_datasets\\digits_processed.csv"))
park_data = pd.read_csv(os.path.join(os.getcwd(), "pre_processed_datasets\\parkinsons_processed.csv"))
#park_data.rename("Diagnosis","label")
rice_data = pd.read_csv(os.path.join(os.getcwd(), "pre_processed_datasets\\rice_processed.csv"))
student_data = pd.read_csv(os.path.join(os.getcwd(), "pre_processed_datasets\\student_dropout_processed.csv"))
#park_data.rename("target","label")
#best_model(digit_data)
#best_model(credit_data)
#best_model(park_data)
#best_model(rice_data)
best_model(student_data)






