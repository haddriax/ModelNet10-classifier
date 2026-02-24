from typing import Any

from .PointNet import PointNet
from .SimplePointNet import SimplePointNet
from .DGCNN import DGCNN
from .PointNetPP import PointNetPP
from .PointTransformer import PointTransformer

ALL_MODELS: dict[str, Any] = {
    "PointNet": PointNet,
    "DGCNN": DGCNN,
    "PointNetPP": PointNetPP,
    "PointTransformer": PointTransformer,
    "SimplePointNet": SimplePointNet,
}
