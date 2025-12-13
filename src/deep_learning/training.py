import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"{self.__class__.__name__}: Using {str(self.device)}")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        from src.deep_learning.models.SimplePointNet import SimplePointNet
        self.model = SimplePointNet(num_classes=10).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for points, labels in tqdm(self.train_loader, desc="Training"):
            points, labels = points.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(points)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), correct / total

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for points, labels in self.test_loader:
                points, labels = points.to(self.device), labels.to(self.device)
                outputs = self.model(points)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            test_acc = self.test()
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    from pathlib import Path
    from src.dataset.dataset import PointCloudDataset

    data_dir = Path("../../data/ModelNet10/models")

    train_ds = PointCloudDataset(data_dir, split='train', n_points=1024)
    test_ds = PointCloudDataset(data_dir, split='test', n_points=1024)

    trainer = ModelTrainer(train_ds, test_ds)
    trainer.train(epochs=10)