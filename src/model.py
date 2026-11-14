"""EmotionNet: a compact 3-block CNN for 5-class facial expression recognition."""
import torch
import torch.nn as nn

CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
IMG_SIZE = (128, 128)


class EmotionNet(nn.Module):
    """Conv(32) -> Conv(64) -> Conv(128), each followed by ReLU + 2x2 max-pool,
    then FC(128) with Dropout(0.5) and a linear classifier head."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        # 128x128 input -> 128 channels of 14x14 after three conv/pool blocks
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)  # raw logits; CrossEntropyLoss applies softmax


def load_model(checkpoint_path: str, device: torch.device) -> EmotionNet:
    """Load trained weights saved by the notebook (best_model.pth / final_emotion_model.pth)."""
    model = EmotionNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model
