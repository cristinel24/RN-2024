import pandas as pd
import os
import torch
from mlp.model import MLPModel

submission_data = {"ID": [], "target": []}

if __name__ == '__main__':
    for _ in range(0, 1):
        model = MLPModel(input_size=784, hidden_size=256, output_size=10, learning_rate=0.001, epochs=100)
        model.train_model()
        model.load_state_dict(torch.load('model.pth'))

        test_data = model.test_data

        for i, (images, _) in enumerate(test_data):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for j, label in enumerate(predicted):
                submission_data["ID"].append(i * test_data.batch_size + j)
                submission_data["target"].append(label.item())

        file_name = f"result_{model.best_accuracy * 100:.2f}_{model.epochs}.csv"
        if not os.path.exists(file_name):
            df = pd.DataFrame(submission_data)
            df.to_csv(file_name, index=False)
