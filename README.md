# mlp-from-scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
#from sklearn.datasets import make_circles

# Dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
#X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

# Κανονικοποίηση
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Train-Test split
np.random.seed(42)
shuffle = np.random.permutation(len(X))
train_size = int(0.8 * len(X))

train_samples = shuffle[:train_size]  # 80% train
test_samples = shuffle[train_size:]   # 20% test

xTrain, yTrain = X[train_samples], y[train_samples]
xTest, yTest = X[test_samples], y[test_samples]

# Activation functions

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)
'''
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)



class MLP:

    def __init__(self, input_size, hidden_layers, output_size):


        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(layer_sizes) - 1

        self.activation = relu
        #self.activation = tanh
        self.activation_deriv = relu_deriv
        #self.activation_deriv = tanh_deriv

        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            rows = layer_sizes[i]
            cols = layer_sizes[i + 1]
            std_dev = (2 / (rows + cols)) ** 0.5
            w = np.random.randn(rows, cols) * std_dev
            self.weights.append(w)

            b = np.zeros((1, cols))
            self.biases.append(b)

    def forward(self, x):
        self.a = [x]
        self.z = []
        for i in range(self.num_layers - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            self.a.append(self.activation(z))

        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        self.a.append(sigmoid(z))  # τελικό επίπεδο sigmoid
        return self.a[-1]

    def backward(self, x, y_true):
        self.forward(x)
        y_true = y_true.reshape(-1, 1)
        y_pred = self.a[-1]

        delta = (y_pred - y_true) * sigmoid_deriv(self.z[-1])

        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers

        grad_w[-1] = np.dot(self.a[-2].T, delta)
        grad_b[-1] = delta

        for l in range(self.num_layers - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.activation_deriv(self.z[l])
            grad_w[l] = np.dot(self.a[l].T, delta)
            grad_b[l] = delta

        return grad_w, grad_b

    def update(self, grad_w, grad_b, lr):
        for i in range(self.num_layers):
            self.weights[i] -= lr * grad_w[i]
            self.biases[i] -= lr * grad_b[i]

    # Εκπαίδευση MLP
    def predict(self, x):
        y_pred = self.forward(x)
        return (y_pred >= 0.5).astype(int)


    def train(self, X, y, epochs=100, lr=0.03, verbose=True):
        losses = []
        accuracies = []

        for epoch in range(epochs):
            correct = 0
            total_loss = 0

            for i in range(len(X)):
                x_sample = X[i].reshape(1, -1)
                y_sample = np.array([y[i]])

                grad_w, grad_b = self.backward(x_sample, y_sample)
                self.update(grad_w, grad_b, lr)

                y_pred = self.a[-1]
                pred_label = (y_pred >= 0.5).astype(int).item()

                if pred_label == y[i]:
                    correct += 1

                loss = binary_crossentropy(y_sample, y_pred)
                total_loss += loss

            avg_loss = total_loss / len(X)
            accuracy = correct / len(X)

            losses.append(avg_loss)
            accuracies.append(accuracy)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        return losses, accuracies

mlp = MLP(input_size=2, hidden_layers=[10, 10], output_size=1)
losses, accuracies = mlp.train(xTrain, yTrain, epochs=100, lr=0.03)

# Plot loss & accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

#  Τest set
y_pred_test = np.array([mlp.predict(x.reshape(1, -1)).item() for x in xTest])
test_acc = np.mean(y_pred_test == yTest)
print(f"\nTest Accuracy: {test_acc:.4f}")

# classification
def classification_report(y_true, y_pred, verbose=True):

    # Υπολογισμός confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))


    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn)


    support_pos = np.sum(y_true == 1)
    support_neg = np.sum(y_true == 0)


    report = {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': accuracy,
        'support': {
            'positive': support_pos,
            'negative': support_neg,
            'total': len(y_true)
        }
    }

    # Εκτύπωση αναφοράς
    if verbose:
        print("\n Enhanced Classification Report")
        print("-" * 40)
        print(f"{'Accuracy':<12}: {accuracy:.4f}")
        print(f"{'Precision':<12}: {precision:.4f}")
        print(f"{'Recall':<12}: {recall:.4f}")
        print(f"{'Specificity':<12}: {specificity:.4f}")
        print(f"{'F1 Score':<12}: {f1:.4f}")
        print("\n Support:")
        print(f"  Positive class: {support_pos}")
        print(f"  Negative class: {support_neg}")
        print(f"  Total samples: {len(y_true)}")


    return report

report = classification_report(yTest, y_pred_test)

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model.predict(p.reshape(1, -1)).item() for p in grid])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(mlp, xTest, yTest)
