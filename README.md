# AI-ML-Assignment-3-Simple-NN
MNIST Digit Classification using a Feedforward Neural Network in TensorFlow/Keras

# MNIST Digit Classification using FNN

**Author:** Denniz Garza  
**Framework:** TensorFlow/Keras

## Project Overview
This project implements a simple Feedforward Neural Network (FNN) to classify handwritten digits from the MNIST dataset. The model predicts digits (0-9) based on the pixel values of each image.

## Model Architecture
- **Input Layer:** 784 neurons (flattened 28x28 images)
- **Hidden Layer 1:** 128 neurons, ReLU activation
- **Hidden Layer 2:** 64 neurons, ReLU activation
- **Output Layer:** 10 neurons, Softmax activation

## Training
- **Loss function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** 10
- **Batch size:** 32
- **Validation split:** 0.1 (10% of training data)

## Evaluation
- **Test Set Accuracy:** ~97%

## Usage
1. Install dependencies: pip install -r requirements.txt
2. Run the notebook `Digit_Dataset.ipynb` to train and evaluate the model.
3. The trained model is saved as `mnist_fnn_model.h5`.

Example evaluation code:
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
