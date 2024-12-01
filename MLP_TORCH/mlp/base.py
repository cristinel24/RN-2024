import torch
from torchvision import datasets, transforms


class BaseModel:
    @staticmethod
    def _download_mnist(is_train: bool):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='./data', train=is_train, transform=transform, download=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=is_train)
        return loader

    @staticmethod
    def _accuracy(predictions, labels):
        _, preds = torch.max(predictions, 1)
        return (preds == labels).float().mean().item()
