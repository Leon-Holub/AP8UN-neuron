import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# y = w1*x1 + w2*x2 + b
'''
example: y = 2*x1 + 3*x2 + 1
    y = 2*1 + 3*2 + 1 = 2 + 6 + 1 = 9
    if y >= 0 -> return 1
    if y < 0 -> return 0
    y -> 1
'''


# error calc -> error = Ytrue - Ypredicated
# Wnew1 = W1 + learning_rate * error * x1 (learning rate could be = 0.01)
# Wnew2 = W2 + learning_rate * error * x2
# bnew = b + learning_rate * error

# When to stop
# 1. set max epoch -> 1000
# 2. SUM(abs(error)) = 0 -> Ypredicated = Ytrue

# startup -> W1, W2, b = 0
# yPredicated = W1*x1 + W2*x2 + b (if yPredicated >= 0 -> return 1, else return 0)
# yTrue = Occupancy from file

def count_error(yTrue, yPred):
    return yTrue - yPred


class Perceptron:
    def __init__(self, resultColumnName, xCount=1, learning_rate=0.01, max_epoch=10000):
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.actual_epoch = 0
        self.weights = [0] * xCount
        self.bias = 0
        self.result_column_name = resultColumnName
        self.errors = []

    def predict(self, rowOfX):
        y = 0
        for i in range(len(rowOfX)):
            y += self.weights[i] * rowOfX.iloc[i]
        y += self.bias
        return 1 if y >= 0 else 0

    def train(self, trainingData):
        for epoch in range(self.max_epoch):
            total_error = 0
            trainingData = trainingData.sample(frac=1).reset_index(drop=True)  # Shuffling dataset
            for _, row in trainingData.iterrows():
                y = self.predict(row[:-1])
                error = count_error(row[self.result_column_name], y)
                self.update_weights(error, row[:-1])
                total_error += abs(error)
            self.errors.append(total_error)
            self.actual_epoch += 1
            if total_error == 0:
                print("No error! I´m done!")
                break

            trainingData = trainingData.sample(frac=1).reset_index(drop=True)  # Shuffling dataset

    def update_weights(self, error, rowOfX):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * rowOfX.iloc[i]
        self.bias += self.learning_rate * error

    def print_weights(self, maxDecimals=3):
        weights_dict = {f"W{i + 1}": round(self.weights[i], maxDecimals) for i in range(len(self.weights))}
        weights_dict["Bias"] = round(self.bias, maxDecimals)

        df = pd.DataFrame(list(weights_dict.items()), columns=["Weight", "Value"])
        print(df)

    def test(self, testData):
        correct = 0
        wrong = 0
        for rowIndex, row in testData.iterrows():
            y = self.predict(row[:-1])
            if y == row[self.result_column_name]:
                correct += 1
            else:
                wrong += 1
        print_test_results(correct, wrong)


def print_test_results(correct, wrong):
    datasetSize = correct + wrong
    data = [
        ["Count", datasetSize, correct, wrong],
        ["Percentage", "", f"{round(correct / datasetSize * 100, 1)}%", f"{round(wrong / datasetSize * 100, 1)}%"]
    ]

    column_labels = ["", "Dataset size", "Correct", "Wrong"]

    fig, ax = plt.subplots(figsize=(5, 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=data, colLabels=column_labels, cellLoc='center', loc='center')

    plt.show()


def load_data_from_csv(fileName):
    return pd.read_csv(fileName)


def split_data(df, resultColumnName, test_size=0.2, random_state=42):
    train, test = train_test_split(df, test_size=test_size, stratify=df[resultColumnName], random_state=random_state)
    return train, test


def print_error_chart(epochs, errors):
    plt.figure(figsize=(20, 10), dpi=100)
    plt.stem(range(epochs), errors, linefmt='b-', markerfmt='bo', basefmt=" ")
    plt.title("Perceptron Training Error Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Total Classification Errors")
    plt.grid(True)
    plt.show()


def show_dataset_as_interactive_plot(data):
    fig = go.Figure(data=[go.Scatter3d(x=data.iloc[:, 0], y=data.iloc[:, 1], z=data.iloc[:, 2], mode='markers',
                                       marker=dict(size=5, color=data.iloc[:, 3], colorscale='Viridis', opacity=0.8))])
    fig.update_layout(scene=dict(xaxis_title=data.columns[0], yaxis_title=data.columns[1], zaxis_title=data.columns[2]))
    fig.show()


def plot_decision_boundary_3D(perceptron, data):
    fig = go.Figure()

    # Přidání bodů datasetu
    fig.add_trace(go.Scatter3d(
        x=data.iloc[:, 0],
        y=data.iloc[:, 1],
        z=data.iloc[:, 2],
        mode='markers',
        marker=dict(size=5, color=data.iloc[:, 3], colorscale='Viridis', opacity=0.8)
    ))

    fig.update_layout(scene=dict(
        xaxis_title=data.columns[0],
        yaxis_title=data.columns[1],
        zaxis_title=data.columns[2],
    ))

    x = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), 100)
    y = np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max(), 100)
    x, y = np.meshgrid(x, y)

    z = (-perceptron.bias - perceptron.weights[0] * x - perceptron.weights[1] * y) / perceptron.weights[2]

    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.5))

    fig.show()


if __name__ == '__main__':
    data = load_data_from_csv('perceptron_dataset.csv')
    show_dataset_as_interactive_plot(data)
    perceptron = Perceptron("Occupancy", xCount=data.shape[1] - 1)
    trainData, testData = split_data(data, perceptron.result_column_name)

    perceptron.train(trainData)
    print_error_chart(perceptron.actual_epoch, perceptron.errors)
    perceptron.print_weights()

    perceptron.test(testData)

    plot_decision_boundary_3D(perceptron, data)
