from typing import Type
from pathlib import Path

from torch import nn

from src.dataset.base_modelnet_dataset import PointCloudDataset
from src.deep_learning.models.SimplePointNet import SimplePointNet
from src.deep_learning.training import ModelTrainer
from src.geometry.Mesh_3D import Sampling

def train_model(model_class: Type[nn.Module],
                epochs: int = 50,
                resume: bool = False,
                batch_size: int = 32,
                data_dir: Path | None = None,
                save_path: Path | None = None,
                ):
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "ModelNet10" / "models"
    if save_path is None:
        save_path = Path(__file__).parent.parent / "models" / f"{model_class.__name__}_modelnet10.pth"

    print(f"Using data directory: {data_dir.resolve()}")

    train_ds = PointCloudDataset(root_dir=data_dir,
                                 split='train',
                                 n_points=1024,
                                 sampling_method=Sampling.UNIFORM,
                                 use_existing_split=True,
                                 cache_processed=True,
                                 )

    test_ds = PointCloudDataset(root_dir=data_dir,
                                split='test',
                                n_points=1024,
                                sampling_method=Sampling.UNIFORM,
                                use_existing_split=True,
                                cache_processed=True)

    num_classes = len(train_ds.class_to_idx)
    model = model_class(num_classes=num_classes)

    trainer = ModelTrainer(
        train_ds,
        test_ds,
        model=model,
        batch_size=batch_size,
        save_model=save_path,
        experiment_name=f"pointnet_baseline_dropout_bs{batch_size}"
    )

    trainer.train(epochs=epochs, resume=resume)

if __name__ == "__main__":
    train_model(model_class=SimplePointNet,
                epochs=50,
                batch_size=256
                )