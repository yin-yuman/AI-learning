import json
import os

import matplotlib.pyplot as plt


# ============================================================
# 1. 获取项目根目录
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

RESULT_PATH = os.path.join(
    RESULT_DIR,
    "residual_experiment.json"
)


# ============================================================
# 2. 读取实验数据
# ============================================================

with open(
    RESULT_PATH,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


# ============================================================
# 3. 找到两个模型的数据
# ============================================================

plain_result = None
resnet_result = None

for result in results:

    if result["model"] == "Plain-CNN":
        plain_result = result

    elif result["model"] == "ResNet":
        resnet_result = result


if plain_result is None or resnet_result is None:

    raise ValueError(
        "没有找到 Plain-CNN 或 ResNet 的实验结果。"
    )


# ============================================================
# 4. Epoch
# ============================================================

epochs = list(
    range(
        1,
        len(plain_result["train_loss"]) + 1
    )
)


# ============================================================
# 5. 输出实验结果
# ============================================================

print("=" * 60)

print("Residual Experiment Results")

print("=" * 60)

print()

print(
    f"Plain-CNN | "
    f"Parameters: "
    f"{plain_result['parameters']:,} | "
    f"Test Accuracy: "
    f"{plain_result['test_accuracy']:.2f}%"
)

print(
    f"ResNet    | "
    f"Parameters: "
    f"{resnet_result['parameters']:,} | "
    f"Test Accuracy: "
    f"{resnet_result['test_accuracy']:.2f}%"
)

print()

print(
    f"Plain-CNN Training Time: "
    f"{plain_result['training_time']:.2f} s"
)

print(
    f"ResNet Training Time: "
    f"{resnet_result['training_time']:.2f} s"
)


# ============================================================
# 6. 图1：Training Loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    plain_result["train_loss"],
    marker="o",
    linewidth=2,
    label="Plain-CNN"
)

plt.plot(
    epochs,
    resnet_result["train_loss"],
    marker="o",
    linewidth=2,
    label="ResNet"
)

plt.xlabel("Epoch")

plt.ylabel("Training Loss")

plt.title(
    "Training Loss Comparison: Plain-CNN vs ResNet"
)

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    RESULT_DIR,
    "residual_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.show()


# ============================================================
# 7. 图2：Training Accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    plain_result["train_accuracy"],
    marker="o",
    linewidth=2,
    label="Plain-CNN"
)

plt.plot(
    epochs,
    resnet_result["train_accuracy"],
    marker="o",
    linewidth=2,
    label="ResNet"
)

plt.xlabel("Epoch")

plt.ylabel("Training Accuracy (%)")

plt.title(
    "Training Accuracy Comparison: Plain-CNN vs ResNet"
)

plt.xticks(epochs)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    RESULT_DIR,
    "residual_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.show()


# ============================================================
# 8. 图3：Test Accuracy
# ============================================================

models = [
    "Plain-CNN",
    "ResNet"
]

test_accuracies = [

    plain_result["test_accuracy"],

    resnet_result["test_accuracy"]

]


plt.figure(figsize=(7, 5))

bars = plt.bar(
    models,
    test_accuracies
)

plt.xlabel("Model")

plt.ylabel("Test Accuracy (%)")

plt.title(
    "Test Accuracy Comparison"
)

plt.ylim(
    0,
    100
)


# 在柱子上显示具体数值

for bar, accuracy in zip(
    bars,
    test_accuracies
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height()
        + 1,

        f"{accuracy:.2f}%",

        ha="center"

    )


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

test_accuracy_path = os.path.join(
    RESULT_DIR,
    "residual_test_accuracy.png"
)

plt.savefig(
    test_accuracy_path,
    dpi=300
)

plt.show()


# ============================================================
# 9. 图4：参数量
# ============================================================

parameters = [

    plain_result["parameters"],

    resnet_result["parameters"]

]


plt.figure(figsize=(7, 5))

bars = plt.bar(
    models,
    parameters
)

plt.xlabel("Model")

plt.ylabel("Number of Parameters")

plt.title(
    "Parameter Comparison"
)


for bar, parameter in zip(
    bars,
    parameters
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{parameter:,}",

        ha="center",

        va="bottom"

    )


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

parameter_path = os.path.join(
    RESULT_DIR,
    "residual_parameters.png"
)

plt.savefig(
    parameter_path,
    dpi=300
)

plt.show()


# ============================================================
# 10. 图5：训练时间
# ============================================================

training_times = [

    plain_result["training_time"],

    resnet_result["training_time"]

]


plt.figure(figsize=(7, 5))

bars = plt.bar(
    models,
    training_times
)

plt.xlabel("Model")

plt.ylabel("Training Time (seconds)")

plt.title(
    "Training Time Comparison"
)


for bar, training_time in zip(
    bars,
    training_times
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{training_time:.1f}s",

        ha="center",

        va="bottom"

    )


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

time_path = os.path.join(
    RESULT_DIR,
    "residual_training_time.png"
)

plt.savefig(
    time_path,
    dpi=300
)

plt.show()


# ============================================================
# 11. 完成
# ============================================================

print()

print("=" * 60)

print("Visualization finished.")

print("=" * 60)

print()

print("Generated files:")

print(
    "1.",
    loss_path
)

print(
    "2.",
    accuracy_path
)

print(
    "3.",
    test_accuracy_path
)

print(
    "4.",
    parameter_path
)

print(
    "5.",
    time_path
)