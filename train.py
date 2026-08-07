import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNN
import matplotlib.pyplot as plt
#图片转换
transform =transforms.ToTensor()
#加载训练集
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform
)
#数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    #随机打乱图片顺序
    shuffle=True
)
model = CNN()
model.train()
criterion=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001)
print(model)
loss_list=[]
for epoch in range(10):

    total_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)

    print(
        f"Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.4f}"
    )
torch.save(
    model.state_dict(),
    'cnn_cifar10.pth')
plt.plot(loss_list)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.savefig("loss_curve.png")

plt.show()