import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('aerialmodel.pth')
model.eval()

