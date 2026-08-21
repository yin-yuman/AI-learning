# ViT论文学习笔记
## 1.论文基本信息
标题：AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

作者：Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
## 2. 研究背景
传统图像识别主要依赖 CNN，通过卷积提取图像特征。Transformer 在自然语言处理领域取得成功后，研究者开始探索其在计算机视觉中的应用。

本文尝试摆脱 CNN 的限制，将 Transformer 直接应用于图像分类。
## 3. 研究问题
本文主要研究： Transformer 能否不依赖 CNN，直接用于图像分类？

同时探究模型规模和训练数据规模对 ViT 性能的影响。
## 4. 核心思想
把一张图片切成一个个固定大小的 Patch，然后把每个 Patch 当成一个 Token，输入 Transformer
整体流程：

Image -> Patch -> Patch Embedding -> Position Embedding -> Transformer Encoder -> Classification
## 5. 方法详解
ViT 的主要方法是将图像转换成 Patch 序列，再利用 Transformer 进行特征提取。

### 5.1 图像切分

首先将输入图像划分为大小相同的 Patch。例如输入一张 224×224 的图片，使用 16×16 的 Patch，可以得到：
14×14=196个 Patch。

### 5.2 Patch Embedding

将每个 Patch 展平并映射到固定维度的向量空间，使其成为 Transformer 可以处理的 Token。

### 5.3 加入 CLS Token 和位置编码

在 Patch 序列前加入一个 CLS Token，用于最终的图像分类。

同时加入 Position Embedding，使模型能够获得不同 Patch 在图像中的位置信息。

### 5.4 Transformer Encoder

处理后的序列输入 Transformer Encoder。

Encoder 主要由：

Multi-Head Self-Attention

MLP

Layer Normalization

Residual Connection

组成。

Self-Attention 可以建立不同 Patch 之间的联系，使模型能够学习图像中的全局信息。

### 5.5 分类

经过多层 Transformer Encoder 后，取出 CLS Token 的特征，输入 MLP Head，最终得到图像的分类结果。
## 6. 模型结构
ViT主要由三部分组成：

Patch Embedding -> Transformer Encoder -> MLP Head

其中 Transformer Encoder 是模型的核心，通过 Multi-Head Self-Attention 建立不同 Patch 之间的关系。
## 7. 实验分析
论文主要通过不同规模的数据集和不同模型规模进行实验，分析 ViT 的图像分类能力。

### 7.1 与 CNN 模型对比

实验将 ViT 与当时先进的 CNN 模型进行比较。

结果表明，在大规模数据预训练的情况下，ViT 可以取得非常优秀的分类性能，说明 Transformer 不依赖 CNN 也能够有效学习图像特征。

### 7.2 数据规模的影响

实验发现，训练数据规模对 ViT 的性能影响非常明显。

在较小的数据集上，ViT 的优势并不明显；随着预训练数据规模增加，ViT 的性能不断提升，并逐渐超过传统 CNN 模型。

这说明 ViT 对大规模数据具有较强的利用能力。

### 7.3 模型规模的影响

论文还比较了不同规模的 ViT 模型。

总体来看：

模型规模越大、预训练数据越充分，ViT 的性能通常越好。

这也说明 Transformer 架构具有较好的扩展能力。

### 7.4 实验结论

实验最终证明：

ViT 可以直接将 Transformer 应用于图像分类，并在大规模数据预训练的条件下取得优秀的性能。

同时也说明，大规模数据和模型扩展是 ViT 发挥优势的重要条件。
## 8. 创新点
将 Transformer 直接应用于图像分类

将图像 Patch 类比为 NLP 中的 Token

减少对 CNN 视觉先验的依赖

证明大规模预训练对视觉 Transformer 的重要性
## 9. 局限性
对训练数据规模要求较高

Self-Attention 计算量较大

缺少 CNN 的局部特征先验
## 10. 个人思考
ViT最大的创新并不是简单地改变网络结构，而是改变了处理图像的方式：不再依赖卷积提取特征，而是将图像转换为 Patch 序列交给 Transformer 处理。
