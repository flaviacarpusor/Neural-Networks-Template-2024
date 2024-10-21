import numpy as np
from mnist_data import download_mnist

# 1: Load MNIST Data
train_inputs, train_labels = download_mnist(True)
test_inputs, test_labels = download_mnist(False)

# convert the data to NumPy arrays
train_inputs = np.array(train_inputs)
train_labels = np.array(train_labels)
test_inputs = np.array(test_inputs)
test_labels = np.array(test_labels)

#  -Normalize Data
train_inputs = train_inputs / 255.0
test_inputs = test_inputs / 255.0  #ca sa fie in intervalul [0, 1]

#    -convert the labels to one-hot-encoding
def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

#  Initialize Weights and Biases
input_size = 784  # Number of input neurons (28x28 pixels)
output_size = 10  # Number of output classes (digits 0-9)

np.random.seed(42)  # For consistency
weights = np.random.randn(input_size, output_size) * 0.01
bias = np.zeros((output_size,))

#  Softmax Function
def softmax(linear_output):
    exp_output = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))  # subtr the max value from each row => stability
    return exp_output / exp_output.sum(axis=1, keepdims=True) #probab

#  Forward Propagation
def forward_propagation(inputs, weights, bias):
    linear_output = np.dot(inputs, weights) + bias
    predictions = softmax(linear_output)
    return predictions


#  Backward Propagation with Gradient Descent
def backward_propagation(inputs, labels, predictions, weights, bias, learning_rate=0.01):
    error = labels - predictions
    weights += learning_rate * np.dot(inputs.T, error) / inputs.shape[0]#average adjustment for each example in the batch
    bias += learning_rate * np.sum(error, axis=0) / inputs.shape[0]

    return weights, bias

#  Training the Perceptron
def train_perceptron(train_inputs, train_labels, weights, bias, epochs=np.random.randint(50, 501), batch_size=100, learning_rate=0.01):
    for epoch in range(epochs):
        permutation = np.random.permutation(train_inputs.shape[0])
        train_inputs_shuffled = train_inputs[permutation]
        train_labels_shuffled = train_labels[permutation]

        for i in range(0, train_inputs.shape[0], batch_size):
            inputs_batch = train_inputs_shuffled[i:i+batch_size]
            labels_batch = train_labels_shuffled[i:i+batch_size]

            # Forward propagation
            predictions = forward_propagation(inputs_batch, weights, bias)

            # Backward propagation
            weights, bias = backward_propagation(inputs_batch, labels_batch, predictions, weights, bias, learning_rate)

        # Calculate accuracy after each epoch
        predictions_train = forward_propagation(train_inputs, weights, bias)
        y_pred_train = np.argmax(predictions_train, axis=1)
        y_true_train = np.argmax(train_labels, axis=1)
        accuracy_train = np.mean(y_pred_train == y_true_train)

        print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {accuracy_train:.4f}")

    return weights, bias

#  Evaluate the Model on Test Data
def evaluate_model(test_inputs, test_labels, weights, bias):
    predictions_test = forward_propagation(test_inputs, weights, bias)
    y_pred_test = np.argmax(predictions_test, axis=1)
    y_true_test = np.argmax(test_labels, axis=1)
    accuracy_test = np.mean(y_pred_test == y_true_test)
    print(f"Test Accuracy: {accuracy_test:.4f}")

# Train and Evaluate the Perceptron
weights, bias = train_perceptron(train_inputs, train_labels, weights, bias, epochs=np.random.randint(50, 501), batch_size=100, learning_rate=0.01)
evaluate_model(test_inputs, test_labels, weights, bias)
