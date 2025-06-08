import torch
from torchvision.datasets import VOCDetection
from torchvision import transforms

def get_voc_dataloader(data_dir="data", batch_size=4):
    transform = transforms.ToTensor()
    dataset = VOCDetection(data_dir, year='2012', image_set='train', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
