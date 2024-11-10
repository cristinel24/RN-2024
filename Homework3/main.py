from mlp.model import MLPModel

if __name__ == '__main__':
    model = MLPModel(input_size=784, hidden_size=100, output_size=10, epochs=500)

    accuracy = model.train(batch_size=100)
    print(f"Accuracy: {accuracy * 100}%")
