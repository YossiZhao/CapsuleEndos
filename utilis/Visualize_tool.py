



import torch
from torch.utils.tensorboard import SummaryWriter

from models.VGG16 import VGG16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model =VGG16().to(device)

writer = SummaryWriter('logs')
writer.add_graph(model)
writer.close()
