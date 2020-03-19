import torch
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InceptionResnetV1()
model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])
print("final")
