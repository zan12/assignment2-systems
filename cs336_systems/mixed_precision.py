import torch
from torch import nn


def mixed_precision_accumulation():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
    
def benchmark_toy_model(model: nn.Module, dtype: torch.dtype, x: torch.Tensor):
    with torch.autocast(device_type="cuda", dtype=dtype):
        y = model(x)
        loss = torch.mean(y)
        loss.backward()


if __name__ == "__main__":
    batch_size, in_features, out_features = 8, 10, 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((batch_size, in_features)).to(device)
    model = ToyModel(in_features, out_features).to(device)
    benchmark_toy_model(model, torch.bfloat16, x)
