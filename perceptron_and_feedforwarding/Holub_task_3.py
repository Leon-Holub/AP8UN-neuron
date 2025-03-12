import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.src.layers import Dense, Dropout
from torch.utils.data import DataLoader, TensorDataset
from keras import Sequential, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score


def load_and_preprocess_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(set(y))}")

    X_perceptron, y_perceptron = X[y < 2], y[y < 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_perceptron, y_perceptron, test_size=0.3,
                                                                random_state=42)

    scaler = StandardScaler()
    return {
        "X_train": scaler.fit_transform(X_train), "X_test": scaler.transform(X_test),
        "y_train": y_train, "y_test": y_test,
        "X_train_p": scaler.fit_transform(X_train_p), "X_test_p": scaler.transform(X_test_p),
        "y_train_p": y_train_p, "y_test_p": y_test_p
    }


def train_perceptron(X_train_p, X_test_p, y_train_p, y_test_p):
    model = Perceptron()
    model.fit(X_train_p, y_train_p)
    y_pred = model.predict(X_test_p)
    print("Scikit-learn Perceptron:")
    print(classification_report(y_test_p, y_pred))


def train_sklearn_feedforward(X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Scikit-learn Feedforward NN:")
    print(classification_report(y_test, y_pred))


def train_keras_feedforward(X_train, X_test, y_train, y_test):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(10, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Keras Feedforward NN:")
    print(classification_report(y_test, y_pred))


class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation_fn
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


def train_pytorch_feedforward(X_train, X_test, y_train, y_test, activation_functions, optimizers):
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_train_torch, y_train_torch)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for activation in activation_functions:
        for optimizer_type in optimizers:
            model = FeedforwardNN(X_train.shape[1], 10, 3, activation)
            criterion = nn.CrossEntropyLoss()

            if optimizer_type == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=0.01)
            elif optimizer_type == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            else:
                raise ValueError(f'Unknown optimizer: {optimizer_type}')

            for epoch in range(30):
                for batch in dataloader:
                    inputs, labels = batch
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            y_pred = torch.argmax(model(X_test_torch), axis=1).detach().cpu().numpy()
            y_true = y_test_torch.detach().cpu().numpy()

            print(f"PyTorch Feedforward NN (Activation: {activation.__class__.__name__}, Optimizer: {optimizer_type}):")
            print(classification_report(y_true, y_pred))


data = load_and_preprocess_data()
train_perceptron(data['X_train_p'], data['X_test_p'], data['y_train_p'], data['y_test_p'])
train_sklearn_feedforward(data['X_train'], data['X_test'], data['y_train'], data['y_test'])
train_keras_feedforward(data['X_train'], data['X_test'], data['y_train'], data['y_test'])
train_pytorch_feedforward(data['X_train'], data['X_test'], data['y_train'], data['y_test'],
                          [nn.ReLU(), nn.Sigmoid(), nn.Tanh()],
                          ['Adam', 'SGD'])
