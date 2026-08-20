import os
import json
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from models.plain_cnn import PlainCNN
from models.resnet import ResNetSmall


# 实验参数

BATCH_SIZE = 64

EPOCHS = 15

LEARNING_RATE = 0.05

TRAIN_SIZE = 20000

TEST_SIZE = 5000


#  设备

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)



BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


#  数据预处理

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

])


#  加载 CIFAR-10

train_dataset = torchvision.datasets.CIFAR10(

    root=DATA_DIR,

    train=True,

    download=False,

    transform=transform

)


test_dataset = torchvision.datasets.CIFAR10(

    root=DATA_DIR,

    train=False,

    download=False,

    transform=transform

)


#  使用部分数据

train_dataset = Subset(

    train_dataset,

    range(TRAIN_SIZE)

)


test_dataset = Subset(

    test_dataset,

    range(TEST_SIZE)

)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Testing samples:",
    len(test_dataset)
)


#  参数量统计

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


#  单个模型训练

def train_model(
    model,
    model_name
):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=LEARNING_RATE,

        momentum=0.9,

        weight_decay=5e-4

    )


    train_losses = []

    train_accuracies = []

    best_accuracy = 0.0


    start_time = time.time()



    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            # Forward

            outputs = model(images)


            # Loss

            loss = criterion(
                outputs,
                labels
            )


            # Backward

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # Loss统计

            running_loss += loss.item()


            # Accuracy统计

            _, predicted = torch.max(

                outputs,

                1

            )


            total += labels.size(0)

            correct += (

                predicted == labels

            ).sum().item()


        epoch_loss = (

            running_loss /
            len(train_loader)

        )


        epoch_accuracy = (

            100.0 *
            correct /
            total

        )


        train_losses.append(
            epoch_loss
        )

        train_accuracies.append(
            epoch_accuracy
        )


        print(

            f"{model_name} | "

            f"Epoch "
            f"{epoch + 1}/{EPOCHS} | "

            f"Loss: "
            f"{epoch_loss:.4f} | "

            f"Accuracy: "
            f"{epoch_accuracy:.2f}%"

        )


        if epoch_accuracy > best_accuracy:

            best_accuracy = epoch_accuracy


    training_time = (
        time.time() - start_time
    )


    return {

        "model": model,

        "loss": train_losses,

        "accuracy": train_accuracies,

        "best_train_accuracy":
            best_accuracy,

        "training_time":
            training_time

    }


#  测试模型

def evaluate_model(model):

    model.eval()

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(

                outputs,

                1

            )


            total += labels.size(0)

            correct += (

                predicted == labels

            ).sum().item()


    accuracy = (

        100.0 *
        correct /
        total

    )


    return accuracy


#  开始实验

model_configs = {

    "Plain-CNN": PlainCNN,

    "ResNet": ResNetSmall

}


results = []


for model_name, model_class in model_configs.items():

    print()
    print("=" * 60)
    print(
        f"Starting experiment: {model_name}"
    )
    print("=" * 60)


    # 创建模型

    model = model_class()


    parameters = count_parameters(
        model
    )


    print(
        f"Parameters: "
        f"{parameters:,}"
    )


    # 训练

    train_result = train_model(

        model,

        model_name

    )


    model = train_result["model"]


    # 测试

    test_accuracy = evaluate_model(
        model
    )


    # 保存模型

    model_filename = (

        model_name.lower()
        .replace("-", "_")
        + "_fair.pth"

    )


    model_path = os.path.join(

        CHECKPOINT_DIR,

        model_filename

    )


    torch.save(

        model.state_dict(),

        model_path

    )


    # 保存结果

    result = {

        "model":
            model_name,

        "parameters":
            parameters,

        "train_loss":
            train_result["loss"],

        "train_accuracy":
            train_result["accuracy"],

        "best_train_accuracy":
            train_result[
                "best_train_accuracy"
            ],

        "test_accuracy":
            test_accuracy,

        "training_time":
            train_result[
                "training_time"
            ]

    }


    results.append(result)


    print()

    print(
        f"{model_name} Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Training Time: "
        f"{train_result['training_time']:.2f} seconds"
    )


#  保存实验结果

result_path = os.path.join(

    RESULT_DIR,

    "residual_experiment_fair.json"

)


with open(

    result_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        results,

        f,

        indent=4

    )


#  输出最终结果

print()
print("=" * 60)
print("Residual Experiment Finished")
print("=" * 60)


for result in results:

    print(

        f"{result['model']:10s} | "

        f"Parameters: "
        f"{result['parameters']:,} | "

        f"Test Accuracy: "
        f"{result['test_accuracy']:.2f}%"

    )


print()
print(
    "Results saved to:"
)

print(
    result_path
)