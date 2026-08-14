import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
transform=transforms.ToTensor()
test_dataset=torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)
test_loader=DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
)
model=CNN()
model.load_state_dict(torch.load("cnn_cifar10.pth"))
model.eval()

correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
accuracy=100*correct/total
print(
    f"Test Accuracy: {accuracy:.2f}%"
)
