import torch
import torch.nn as nn
import torch.nn.functional as F

class InputTransformNet(nn.Module):
    def __init__(self, K=3):
        super(InputTransformNet, self).__init__()
        self.K = K
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.conv3 = nn.Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K*K)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch_size, 1, num_point, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)  # (batch_size, 1024, 1, 1)
        x = x.reshape(batch_size, -1)  # (batch_size, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch_size, K*K)
        x = x.reshape(batch_size, self.K, self.K)  # (batch_size, 3, 3)
        return x

class FeatureTransformNet(nn.Module):
    def __init__(self, K=64):
        super(FeatureTransformNet, self).__init__()
        self.K = K
        self.conv1 = nn.Conv2d(K, 64, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.conv3 = nn.Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K*K)
        self.fc3.weight.data.zero_()
        identity = torch.eye(K).flatten()
        self.fc3.bias.data = identity
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)  # (batch_size, 1024, 1, 1)
        x = x.reshape(batch_size, -1)  # (batch_size, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch_size, K*K)
        x = x.reshape(batch_size, self.K, self.K)  # (batch_size, 64, 64)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes: int = 10, input_K=3, feature_K=64):
        super(PointNet, self).__init__()
        self.input_transform = InputTransformNet(K=input_K)
        self.feature_transform = FeatureTransformNet(K=feature_K)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.conv5 = nn.Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1), padding='valid')
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        batch_size, num_point, _ = x.shape

        # Input transform
        transform = self.input_transform(x)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_point)
        x = torch.bmm(transform, x)  # (batch_size, 3, num_point)
        x = x.transpose(2, 1)  # (batch_size, num_point, 3)

        # PointNet backbone
        x = x.unsqueeze(1)  # (batch_size, 1, num_point, 3)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, num_point, 1)

        # Feature transform
        # x is (batch_size, 64, num_point, 1) — pass directly to FeatureTransformNet
        transform = self.feature_transform(x)  # (batch_size, 64, 64)
        # Apply transform: x[B, N, 64] @ transform[B, 64, 64] -> [B, N, 64]
        x = x.squeeze(-1).transpose(1, 2)  # (batch_size, num_point, 64)
        x = x.reshape(batch_size * num_point, 1, 64)  # (batch_size*num_point, 1, 64)
        transform_exp = transform.unsqueeze(1).expand(-1, num_point, -1, -1)  # (batch_size, num_point, 64, 64)
        transform_exp = transform_exp.reshape(batch_size * num_point, 64, 64)  # (batch_size*num_point, 64, 64)
        x = torch.bmm(x, transform_exp)  # (batch_size*num_point, 1, 64)
        x = x.reshape(batch_size, num_point, 64).transpose(1, 2).unsqueeze(-1)  # (batch_size, 64, num_point, 1)

        # Continue with remaining layers
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.maxpool(x)  # (batch_size, 1024, 1, 1)
        x = x.reshape(batch_size, -1)  # (batch_size, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Test
if __name__ == '__main__':
    batch_size = 32
    num_point = 512
    model = PointNet(num_classes=10)

    # Create random input
    inputs = torch.randn((batch_size, num_point, 3))

    try:
        outputs = model(inputs)
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")  # Should be (32, 10)
    except Exception as e:
        print(f"Error: {e}")
