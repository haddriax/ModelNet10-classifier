import time
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime


class TrainingResults(TypedDict):
    """Results returned by ModelTrainer.train()"""
    best_test_acc: float
    final_train_acc: float
    final_train_loss: float
    final_test_loss: float
    final_test_acc: float
    per_class_accuracies: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    model_path: str
    best_model_path: str
    total_training_time_seconds: float
    epochs_trained: int


class ModelTrainer:
    def __init__(self, train_dataset: Dataset,
                 test_dataset: Dataset,
                 model: nn.Module,
                 save_model: Path,
                 batch_size: int = 32,
                 experiment_name: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"{self.__class__.__name__}: Using {str(self.device)}")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.save_model_path = save_model

        # Create parent directory if it doesn't exist
        self.save_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True

        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True
        )

        base_lr = 0.001
        self.lr = base_lr * (batch_size / 32)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # TensorBoard setup
        if experiment_name is None:
            experiment_name = f"pointnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_dir = Path("runs") / experiment_name
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
        print(f"Run: tensorboard --logdir=runs")

        # Track best accuracy for saving
        self.best_test_acc = 0.0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (points, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            points, labels = points.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(points)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # Log batch loss every 10 batches
            if batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/batch', loss.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def test(self, epoch: int) -> dict:
        """Evaluate model on test set.

        Returns:
            Dict with keys: loss, accuracy, per_class_accuracies,
            per_class_precision, per_class_recall, per_class_f1,
            macro_precision, macro_recall, macro_f1
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0

        num_classes = len(self.test_dataset.class_to_idx)
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        class_predicted = [0] * num_classes

        with torch.no_grad():
            for points, labels in tqdm(self.test_loader, desc="Testing"):
                points, labels = points.to(self.device), labels.to(self.device)
                outputs = self.model(points)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                predictions = outputs.argmax(1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                for label, prediction in zip(labels, predictions):
                    class_total[label] += 1
                    class_predicted[prediction] += 1
                    if label == prediction:
                        class_correct[label] += 1

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        # Per-class metrics
        per_class_acc: dict[str, float] = {}
        per_class_precision: dict[str, float] = {}
        per_class_recall: dict[str, float] = {}
        per_class_f1: dict[str, float] = {}

        for i in range(num_classes):
            class_name = self.test_dataset.get_class_name(i)

            acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            precision = class_correct[i] / class_predicted[i] if class_predicted[i] > 0 else 0.0
            recall = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_acc[class_name] = acc
            per_class_precision[class_name] = precision
            per_class_recall[class_name] = recall
            per_class_f1[class_name] = f1

            self.writer.add_scalar(f'Accuracy/class_{class_name}', acc, epoch)
            self.writer.add_scalar(f'Precision/class_{class_name}', precision, epoch)
            self.writer.add_scalar(f'Recall/class_{class_name}', recall, epoch)
            self.writer.add_scalar(f'F1/class_{class_name}', f1, epoch)

        # Macro averages
        n = len(per_class_precision)
        macro_precision = sum(per_class_precision.values()) / n if n > 0 else 0.0
        macro_recall = sum(per_class_recall.values()) / n if n > 0 else 0.0
        macro_f1 = sum(per_class_f1.values()) / n if n > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "per_class_accuracies": per_class_acc,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    def save_checkpoint(self, epoch: int, test_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'test_acc': test_acc,
            'best_test_acc': self.best_test_acc,
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.save_model_path)

        # Save best model separately
        if is_best:
            best_path = self.save_model_path.parent / f"{self.save_model_path.stem}_best{self.save_model_path.suffix}"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: Path = None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.save_model_path

        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_test_acc = checkpoint.get('best_test_acc', 0.0)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def train(self, epochs: int = 10, resume: bool = False) -> TrainingResults:
        start_time = time.time()
        start_epoch = 0

        # Resume training if requested
        if resume:
            start_epoch = self.load_checkpoint()

        # Log model graph (only once)
        try:
            dummy_input = torch.randn(1, 1024, 3).to(self.device)
            self.writer.add_graph(self.model, dummy_input)
        except Exception as e:
            print(f"Could not log model graph: {e}")

        final_train_loss, final_train_acc = 0.0, 0.0
        final_test_metrics: dict = {}

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            test_metrics = self.test(epoch)

            # Track final epoch values
            final_train_loss, final_train_acc = train_loss, train_acc
            final_test_metrics = test_metrics

            test_loss = test_metrics["loss"]
            test_acc = test_metrics["accuracy"]

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc, epoch)
            self.writer.add_scalar('Precision/macro', test_metrics["macro_precision"], epoch)
            self.writer.add_scalar('Recall/macro', test_metrics["macro_recall"], epoch)
            self.writer.add_scalar('F1/macro', test_metrics["macro_f1"], epoch)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_rate', current_lr, epoch)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"F1: {test_metrics['macro_f1']:.4f}")

            # Save checkpoint
            is_best = test_acc > self.best_test_acc
            if is_best:
                self.best_test_acc = test_acc

            self.save_checkpoint(epoch, test_acc, is_best)

        total_time = time.time() - start_time
        best_model_path = (self.save_model_path.parent /
                           f"{self.save_model_path.stem}_best{self.save_model_path.suffix}")

        # Build results
        self.results = TrainingResults(
            best_test_acc=self.best_test_acc,
            final_train_acc=final_train_acc,
            final_train_loss=final_train_loss,
            final_test_loss=final_test_metrics.get("loss", 0.0),
            final_test_acc=final_test_metrics.get("accuracy", 0.0),
            per_class_accuracies=final_test_metrics.get("per_class_accuracies", {}),
            per_class_precision=final_test_metrics.get("per_class_precision", {}),
            per_class_recall=final_test_metrics.get("per_class_recall", {}),
            per_class_f1=final_test_metrics.get("per_class_f1", {}),
            macro_precision=final_test_metrics.get("macro_precision", 0.0),
            macro_recall=final_test_metrics.get("macro_recall", 0.0),
            macro_f1=final_test_metrics.get("macro_f1", 0.0),
            model_path=str(self.save_model_path),
            best_model_path=str(best_model_path),
            total_training_time_seconds=total_time,
            epochs_trained=epochs - start_epoch,
        )

        # Final save
        print(f"Training complete. Best test accuracy: {self.best_test_acc:.4f}")
        print(f"Model saved to: {self.save_model_path}")

        self.writer.close()
        return self.results
