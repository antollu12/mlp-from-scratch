Multi-Layer Perceptron Neural Network (From Scratch)

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch using Python and NumPy.
The model is trained to classify synthetic datasets such as the two-moon dataset using backpropagation and gradient descent.

The goal of this project is to demonstrate the internal mechanics of neural networks without relying on deep learning frameworks like TensorFlow or PyTorch.

Features:

Implementation of a fully connected neural network
Forward propagation
Backpropagation algorithm
Binary classification using sigmoid output
Training with binary cross-entropy loss

Visualization of:
training loss,
accuracy,
decision boundary,
Custom classification report (precision, recall, F1-score)

Technologies Used:
Python,
NumPy,
Matplotlib,
Scikit-learn dataset generator

Libraries:
NumPy – numerical computation
Matplotlib – visualization
scikit-learn – dataset generation
How the Model Works

The neural network consists of:
Input Layer: 2 features,
Hidden Layers: 2 layers with 10 neurons each,
Output Layer: 1 neuron with sigmoid activation

Activation functions used:
Hidden layers,
ReLU,
Output layer,
Sigmoid,
Loss function,
Binary Cross Entropy,

Training algorithm:
Gradient Descent,
Backpropagation,
Dataset

The model is trained on a synthetic dataset generated using the two-moons dataset from scikit-learn.

Dataset characteristics:
1000 samples,
Non-linear classification problem,
Noise added for realism

The dataset is split into:
80% training data,
20% testing data

Training Process:
During training the following metrics are recorded:
Loss per epoch,
Accuracy per epoch

These metrics are visualized using plots after training.
Evaluation Metrics

After training, the model evaluates performance using:
Accuracy,
Precision,
Recall,
Specificity,
F1 Score,
A custom classification report is generated.
