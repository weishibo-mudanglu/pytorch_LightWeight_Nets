import torch
import torch.nn as nn

model = nn.Linear(10, 1)
criterion = nn.BCEWithLogitsLoss()

x = torch.randn(16, 10)
y = torch.empty(16).random_(2)  # (16, )

out = model(x)  # (16, 1)
out = out.squeeze(dim=-1)  # (16, )

loss = criterion(out, y)