#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:20:57 2025

@author: elmobarratt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

class Model:
    def __init__(self, featureData, observationData, learningRate=0.01, hidden_layers=[20,20], activation='leaky_relu'):
        # Split train/test (70% train)
        total_samples = featureData.shape[1]
        self.m = int(total_samples * 1)

        self.learningRate = learningRate

        self.trainFeatures = featureData[:, :self.m]
        self.trainObservations = observationData[:, :self.m]
        self.testFeatures = featureData[:, self.m:]
        self.testObservations = observationData[:, self.m:]

        # Standardize inputs and outputs based on training data
        self.feature_mean = np.mean(self.trainFeatures, axis=1, keepdims=True)
        self.feature_std = np.std(self.trainFeatures, axis=1, keepdims=True)
        self.feature_std[self.feature_std == 0] = 1  # avoid div by zero

        self.output_mean = np.mean(self.trainObservations, axis=1, keepdims=True)
        self.output_std = np.std(self.trainObservations, axis=1, keepdims=True)
        self.output_std[self.output_std == 0] = 1

        self.trainFeatures = (self.trainFeatures - self.feature_mean) / self.feature_std
        self.testFeatures = (self.testFeatures - self.feature_mean) / self.feature_std

        self.trainObservations = (self.trainObservations - self.output_mean) / self.output_std
        self.testObservations = (self.testObservations - self.output_mean) / self.output_std

        # Define layer sizes: input + hidden + output
        self.layers = [self.trainFeatures.shape[0]] + hidden_layers + [self.trainObservations.shape[0]]
        self.numLayers = len(self.layers)

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.numLayers - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i+1]
            if activation == 'leaky_relu':
                # He initialization
                w = np.random.randn(fan_out, fan_in) * np.sqrt(2 / fan_in)
            elif activation == 'tanh':
                # Xavier initialization
                w = np.random.randn(fan_out, fan_in) * np.sqrt(1 / fan_in)
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            b = np.zeros((fan_out, 1))
            self.weights.append(w)
            self.biases.append(b)

        # Activation function selection
        self.activation_name = activation

        if activation == 'leaky_relu':
            self.activFunc = self.leaky_relu
            self.dActivFunc = self.d_leaky_relu
        else:
            self.activFunc = self.tanh
            self.dActivFunc = self.d_tanh

        # Containers for forward/backward pass
        self.states = [None] * self.numLayers
        self.activations = [None] * self.numLayers
        self.errors = [None] * (self.numLayers - 1)

        self.costs = []

    # Activation functions
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def d_leaky_relu(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1 - np.tanh(x) ** 2

    # Loss and gradient
    @staticmethod
    def loss(yhat, y):
        return np.mean((yhat - y) ** 2)

    def dLoss(self, yhat, y):
        m = y.shape[1]
        return 2 * (yhat - y) / m

    def feedForward(self, X=None):
        # Use given X or trainFeatures if None
        if X is None:
            X = self.trainFeatures

        self.states[0] = X
        self.activations[0] = X

        for i in range(self.numLayers - 1):
            self.states[i+1] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]

            if i == self.numLayers - 2:  # output layer: linear activation (for regression)
                self.activations[i+1] = self.states[i+1]
            else:
                self.activations[i+1] = self.activFunc(self.states[i+1])

        # Compute cost on training data
        cost = self.loss(self.activations[-1], self.trainObservations)
        self.costs.append(cost)
        return cost

    def backPropagation(self):
        # Compute output error
        self.errors[-1] = self.dLoss(self.activations[-1], self.trainObservations)

        # Backpropagate errors through layers
        for i in reversed(range(self.numLayers - 2)):
            self.errors[i] = np.dot(self.weights[i+1].T, self.errors[i+1]) * self.dActivFunc(self.states[i+1])

        # Update weights and biases
        for i in range(self.numLayers - 1):
            grad_w = np.dot(self.errors[i], self.activations[i].T)
            grad_b = np.sum(self.errors[i], axis=1, keepdims=True)

            self.weights[i] -= self.learningRate * grad_w
            self.biases[i] -= self.learningRate * grad_b
  
    
    def predict(self, X):
        # Standardize input
        X_std = (X - self.feature_mean) / self.feature_std

        self.states[0] = X_std
        self.activations[0] = X_std

        for i in range(self.numLayers - 1):
            self.states[i+1] = np.dot(self.weights[i], self.activations[i]) + self.biases[i]

            if i == self.numLayers - 2:
                self.activations[i+1] = self.states[i+1]
            else:
                self.activations[i+1] = self.activFunc(self.states[i+1])

        # Inverse standardize output
        y_pred_std = self.activations[-1]
        y_pred = y_pred_std * self.output_std + self.output_mean
        return y_pred



    def train(self, epochs=10000, verbose=True, print_every=100):
        for epoch in range(1, epochs + 1):
            cost = self.feedForward()
            self.backPropagation()
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch}: cost = {cost:.6f}")

    

    def train_with_animation(self, x_test, y_test, epochs=10000,  record_every=100):
        predictions = []
        losses = []
        
        for epoch in range(1, epochs + 1):
            model.feedForward()
            model.backPropagation()
            
            if epoch % record_every == 0 or epoch == 1:
                y_pred = model.predict(x_test)
                predictions.append(y_pred.flatten())
                loss = np.mean((y_pred - y_test)**2)
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return predictions, losses





def animate_training(x, y, predictions, interval=200):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, label="True Data", s=10, color='blue', alpha=0.5)
    line, = ax.plot([], [], color='red', linewidth=2, label="Model Prediction")
    ax.legend()
    ax.set_title("Model Predictions Over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        line.set_data(x, predictions[frame])
        ax.set_title(f"Model Prediction (Epoch {frame * 100})")
        return line,
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(predictions),
        init_func=init,
        blit=True,
        interval=interval,
        repeat=False
    )

    plt.show()
    anim.save("/Users/elmobarratt/Desktop/Python/training.gif", writer=PillowWriter(fps=30))





# Generate toy data: y = x^3 - 2x + 1 with noise
np.random.seed(42)
x = np.linspace(0, 100, 300).reshape(1, -1)

def f(x):
    return 1/100000000 * x * (100-x) * (x-25)**2 * (x-80)**2

y_true = f(x)

noise = np.random.randn(*y_true.shape)
y = y_true + noise

# Instantiate and train the model
model = Model(x, y, learningRate=0.01, hidden_layers=[50,100,100,50], activation='leaky_relu')

#%%

predictions, losses = model.train_with_animation(x,y_true,50000)
animate_training(x,y_true,predictions,interval=100)



#%%
model.train(epochs=20000, verbose=True, print_every=300)



# Predict on the whole dataset


# Plot results
plt.figure(figsize=(8,5))
plt.scatter(x.flatten(), y.flatten(), label="Noisy observations", alpha=0.3, s=15)



plt.legend()
plt.title("Model fitting a cubic function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.plot(x.flatten(), y.flatten(), label="True")
plt.plot(x.flatten(), model.predict(x).flatten(), label="Predicted")
plt.legend()
plt.show()