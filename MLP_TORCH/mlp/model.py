import torch
import torch.nn as nn
import torch.optim as optim
from mlp.base import BaseModel


class MLPModel(nn.Module, BaseModel):
    def __init__(self, input_size=784, hidden_size=100, output_size=10, learning_rate=0.001, epochs=20):
        super(MLPModel, self).__init__()
        self.best_accuracy = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.train_data = super()._download_mnist(True)
        self.test_data = super()._download_mnist(False)

    def forward(self, data):
        data = data.view(-1, self.input_size)
        data = self.relu(self.in_layer(data))
        data = self.out_layer(data)
        return data

    def train_model(self):
        for epoch in range(self.epochs):
            running_loss = 0.0

            for images, labels in self.train_data:
                self.optimizer.zero_grad()

                outputs = self(images)
                loss = self.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(self.train_data):.4f}')

            accuracy = self.evaluate()
            print(f"accuracy: {accuracy * 100:.2f}%")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print("New best accuracy achieved, saving model...")
                torch.save(self.state_dict(), 'model.pth')

        print(f'Best Accuracy: {self.best_accuracy * 100:.2f}%')

    def evaluate(self):
        total_accuracy = 0.0

        for images, labels in self.test_data:
            outputs = self(images)
            accuracy = super()._accuracy(outputs, labels)
            total_accuracy += accuracy

        return total_accuracy / len(self.test_data)
