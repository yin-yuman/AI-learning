import torch
import torchvision
import torchvision.transforms as transforms
from model import CNN
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
classes = [
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车"
]
# 图片处理
transform = transforms.Compose([
    transforms.ToTensor()
])
# 加载测试集
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)
# 取一张图片
image, label = test_dataset[0]
# 创建模型
model = CNN()
# 加载训练好的参数
model.load_state_dict(
    torch.load("cnn_cifar10.pth")
)
# 设置预测模式
model.eval()
# 增加batch维度
image = image.unsqueeze(0)
# 不计算梯度
with torch.no_grad():
    output = model(image)
    # 找概率最大的类别
    _, predicted = torch.max(output,1)
print("真实类别：", classes[label])
print("预测类别：", classes[predicted.item()])
plt.imshow(
    image.squeeze(0).permute(1,2,0)
)
plt.title(
    "预测：" + classes[predicted.item()]+
    " 实际：" + classes[label]
)
plt.axis("off")
plt.show()