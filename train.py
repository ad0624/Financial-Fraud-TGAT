import torch
import torch.nn.functional as F
from models.tgat import TGAT
from utils.dataset_builder import build_synthetic_graph

# 使用 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建图数据
data = build_synthetic_graph().to(device)

# 初始化 TGAT 模型
model = TGAT(in_dim=16, hidden_dim=32, out_dim=2).to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练模型
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_time)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = out.argmax(dim=1)
        acc = (pred == data.y).sum().item() / len(data.y)
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")
