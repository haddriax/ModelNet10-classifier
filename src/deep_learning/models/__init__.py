from .SimplePointNet import SimplePointNet
from .DGCNN import DGCNN
from .PointNetPP import PointNetPP
from .PointTransformer import PointTransformer

ALL_MODELS: dict[str, type] = {
    "SimplePointNet": SimplePointNet,
    "DGCNN": DGCNN,
    "PointNetPP": PointNetPP,
    "PointTransformer": PointTransformer,
}
