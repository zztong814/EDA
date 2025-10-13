import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 13)
        )

    def forward(self, x):
        return self.model(x)

# ===== 测试一下 =====
if __name__ == "__main__":
    batch_size = 32
    x = torch.randn(batch_size, 13)
    model = Regressor()
    y = model(x)
    print(y.shape)  # 结果应为 torch.Size([32, 13])
