import time
import numpy as np
from threading import Thread
from queue import Queue
from torchvision.datasets import MNIST
from sklearn.utils import shuffle


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', download=True, train=is_train)

    mnist_data = []
    mnist_labels = []

    for image, label in dataset:
        mnist_data.append(np.array(image).flatten())
        mnist_labels.append(label)

    mnist_data = np.array(mnist_data, dtype=np.float32) / 255.0
    mnist_labels = np.array(mnist_labels, dtype=np.int64)

    return mnist_data, mnist_labels


def initialize_parameters(min, max):
    seed = int(time.time())
    np.random.seed(seed)
    weights = np.random.randn(min, max) * 0.01
    bias = np.zeros(max)
    return weights, bias


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(x, weights, bias):
    z = np.dot(x, weights) + bias
    return softmax(z)


def cross_entropy_loss(predictions, targets):
    m = predictions.shape[0]
    result = -np.log(predictions[range(m), targets])
    loss = np.sum(result) / m
    return loss


def backward_propagation(x, y, predictions, weights, bias, learning_rate):
    m = x.shape[0]
    dz = predictions
    dz[range(m), y] -= 1
    dz /= m

    dW = np.dot(x.T, dz)
    db = np.sum(dz, axis=0)

    weights -= learning_rate * dW
    bias -= learning_rate * db

    return weights, bias


def accuracy(predictions, labels):
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == labels)


def train_batch(queue, weights, bias, learning_rate: int):
    while not queue.empty():
        x_batch, y_batch = queue.get()
        predictions = forward_propagation(x_batch, weights, bias)
        weights, bias = backward_propagation(x_batch, y_batch, predictions, weights, bias, learning_rate)
        queue.task_done()
    return weights, bias


def train_perceptron(train_x, train_y, weights, bias, epochs: int, batch_size: int, learning_rate: float, num_threads: int):
    for epoch in range(epochs):
        train_x, train_y = shuffle(train_x, train_y)

        queue = Queue()
        for i in range(0, train_x.shape[0], batch_size):
            x_batch = train_x[i:i + batch_size]
            y_batch = train_y[i:i + batch_size]
            queue.put((x_batch, y_batch))

        threads = []
        for _ in range(num_threads):
            thread = Thread(target=train_batch, args=(queue, weights, bias, learning_rate))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        predictions = forward_propagation(train_x, weights, bias)
        loss = cross_entropy_loss(predictions, train_y)
        accuracy = evaluate(test_X, test_Y, weights, bias)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    return weights, bias


def evaluate(test_x, test_y, weights, bias):
    predictions = forward_propagation(test_x, weights, bias)
    return accuracy(predictions, test_y)


if __name__ == "__main__":
    train_X, train_Y = download_mnist(True)
    test_X, test_Y = download_mnist(False)

    input_size = 784
    output_size = 10
    W, b = initialize_parameters(input_size, output_size)

    epochs = 150
    learning_rate = 0.02
    batch_size = 100
    num_threads = 8

    initial_accuracy = evaluate(test_X, test_Y, W, b)
    print(f'Initial Test Accuracy: {initial_accuracy * 100:.2f}%')

    W, b = train_perceptron(train_X, train_Y, W, b,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            num_threads=num_threads)

    final_accuracy = evaluate(test_X, test_Y, W, b)
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f}%')
