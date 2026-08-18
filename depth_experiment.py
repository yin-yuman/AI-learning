import os
import json
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from models.vgg import vgg4, vgg6, vgg8

# 实验参数


BATCH_SIZE = 64

EPOCHS = 10

LEARNING_RATE = 0.05

TRAIN_SIZE = 20000

TEST_SIZE = 5000


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)



os.makedirs(
    "../checkpoints",
    exist_ok=True
)

os.makedirs(
    "../results",
    exist_ok=True
)


transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

])



train_dataset = torchvision.datasets.CIFAR10(

    root="../data",

    train=True,

    download=False,

    transform=transform

)


train_dataset = Subset(
    train_dataset,
    range(TRAIN_SIZE)
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


test_dataset = torchvision.datasets.CIFAR10(

    root="../data",

    train=False,

    download=False,

    transform=transform

)


test_dataset = Subset(

    test_dataset,

    range(TEST_SIZE)

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


models = {

    "VGG-4": vgg4,

    "VGG-6": vgg6,

    "VGG-8": vgg8

}


def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )



def train_model(model, model_name):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=LEARNING_RATE,

        momentum=0.9,

        weight_decay=5e-4

    )


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


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            running_loss += loss.item()


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
            100.0 * correct / total
        )


        print(

            f"{model_name} | "

            f"Epoch "
            f"{epoch + 1}/{EPOCHS} | "

            f"Loss: "
            f"{epoch_loss:.4f} | "

            f"Train Accuracy: "
            f"{epoch_accuracy:.2f}%"

        )


        if epoch_accuracy > best_accuracy:

            best_accuracy = epoch_accuracy


    training_time = (
        time.time() - start_time
    )


    return (

        model,

        best_accuracy,

        training_time

    )



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

        100.0 * correct / total

    )


    return accuracy


results = []


for model_name, model_function in models.items():

    print()
    print("=" * 60)
    print(f"Starting experiment: {model_name}")
    print("=" * 60)


    # 创建模型

    model = model_function()


    # 参数量

    parameters = count_parameters(model)


    print(
        f"Parameters: {parameters:,}"
    )


    # 训练

    model, train_accuracy, training_time = (

        train_model(

            model,

            model_name

        )

    )


    # 测试

    test_accuracy = evaluate_model(
        model
    )


    # 保存模型

    model_path = (

        f"../checkpoints/"
        f"{model_name.lower()}_depth.pth"

    )


    torch.save(

        model.state_dict(),

        model_path

    )


    # 保存结果

    result = {

        "model": model_name,

        "parameters": parameters,

        "train_accuracy": train_accuracy,

        "test_accuracy": test_accuracy,

        "training_time": training_time

    }


    results.append(result)


    print()
    print(
        f"{model_name} Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )


# 保存实验结果

result_path = (
    "../results/depth_experiment.json"
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
print("Depth Experiment Finished")
print("=" * 60)


for result in results:

    print(

        f"{result['model']:6s} | "

        f"Parameters: "
        f"{result['parameters']:,} | "

        f"Test Accuracy: "
        f"{result['test_accuracy']:.2f}%"

    )