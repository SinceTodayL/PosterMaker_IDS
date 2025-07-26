主干代码来源：https://github.com/alimama-creative/PosterMaker



# PROCESS



>Before 202507

跑通模型，修改部分代码，不影响任何执行逻辑和模型结构，减少运行时的峰值显存



> 202507 11》

准备添加 IDS 模块，改进 TextrenderNet 

工作：

* 数据集？模型需要学习怎么将汉字映射到 IDS 结构
* 如何训练？训练代码？冻结哪些参数？哪些参数是可训练的？
* 评估方式？包括 损失函数、评估对比方式？

> 核心论点 (Thesis Statement): *“相较于现有的字符级（Character-level）文本渲染方法，我们提出了一种基于表意文字描述序列（IDS）的组件级（Component-level）特征组合网络。该方法不仅显著提升了对结构复杂汉字（unseen complex characters）的零样本生成能力和视觉一致性，还能在保持甚至超越原有场景融合能力的同时，为文本的风格化控制提供了新的范式。”*
>
> 基于此，我们的工作需要证明以下几点：
>
> * 优越性: IDS方法在文本渲染上显著优于字符级方法。
>
> * 兼容性: 我们的改进可以无缝集成到现有的大型文生图模型（如PosterMaker）中，不损害其原有能力。
>
> * 扩展性 (Novelty): 我们的方法为未来更精细的文本风格控制打开了大门。



>202507 12

Step1: 将输入文字查表，转化成标准 IDS 结构

Step2: 用 IDS 结构训练模型， TextRenderNet, Adapter (问题：优化目标是什么？是整个模型一块训练，还是将 TextRenderNet 单独拿出来训练？)

* * 我们想要的通过 IDS 渲染出复杂汉字的方式，是不是一个单独的模型？

-

>202507 18-19

转化为官方数据库，修改文件 `utils/ids_query.py` 

使用方法：

```python
from utils.ids_query import IDSQuery
ids_query = IDSQuery()
result = ids_query.query_text(user_input)
```

输出类似于：

```txt
橋 
[{'橋': {'simple': '⿰木喬', 'recursive': '⿰⿻⿻一丨𠆢⿱⿱⿱㇒⿻一人口⿵冂口'}}]
单
[{'单': {'simple': '⿱丷⿻甲一', 'recursive': '⿱⿰丶㇒⿻⿻⿴囗一丨一'}}]
```

simple 是只查找一步，recursive 是递归查找直到最简

一个结构有多种分解方式的时候，默认采用第一种

通过 `query_text` 方法调用的时候，会将英文字母全部忽略掉，输出没有英文字母，如果后面需要改，可以直接加

```python
for char in text:
     if '\u4e00' <= char <= '\u9fff':  # Chinese characters only
            # QUERY
```



>202507 21-26

在将向量接入TextRenderNet之前，按照 PosterMaker 的训练方法

难点主要有：

1. **IDS→视觉对齐**：IDS 本身是符号序列，并不包含笔画粗细、衬线／非衬线等风格信息；要训练一个编码器把结构信息映射到与视觉特征同分布的向量。
2. **歧义与多解**：同一字符在 IDS 上可能有多种分解；需选定规范或注入可学习的“权重”让模型自动消解歧义。



构建IDS词表和tokenizer - 解析所有IDS字符(⿰⿱等)和部件，创建token映射

实现IDS数据流水线 - 将文本转换为IDS token序列，集成到训练数据pipeline

设计IDS-TextRenderNet架构 - 基于现有TextRenderNet改造，支持IDS序列输入

准备IDS训练数据 - 生成IDS级别的字体渲染数据，建立supervision机制

训练IDS-TextRenderNet - 实现训练循环，调试损失函数和优化策略

集成到PosterMaker - 修改推理流程，支持IDS和字符级模型切换

评估和对比 - 在FID/CLIP-Score/OCR准确率等指标上对比原方法



| Token   | 编号 | 含义     | 用途简述           |
| ------- | ---- | -------- | ------------------ |
| `<PAD>` | 0    | 填充符   | 对齐长度           |
| `<BOS>` | 1    | 句子开始 | 开始生成           |
| `<EOS>` | 2    | 句子结束 | 停止生成           |
| `<UNK>` | 3    | 未知词   | 替代词表外的词     |
| `<SEP>` | 4    | 分隔符   | 分隔多个片段或字符 |



> ids 部分

1. IDS编码: "青" → [CLS, ⿱, 艹, 丹, SEP] + token_type_ids + attention_mask
2. Embedding: 256维IDS特征 → 64维投影
3. 压缩/填充: 变长序列 → 固定16个token（weighted pooling）
4. 字符位置编码: 添加32维sinusoidal编码
5. 文本位置编码: 添加32维Fourier编码（基于bounding box）
6. 拼接: 64+32+32=128维 → 扩展到1472维
7. 最终输出: (112, 1472) [7个文本×16个token×1472维]



通过 Adapter 进行特征对齐



目前添加的：

* `utils/ids_query.py` 实现 IDSQuery 类，可以将汉字/句子转换为 ids 表示，有一步分解/递归分解

* `utils/ids_tokenizer.py` 实现 IDSTokenizer 类，构建词表 `assets/ids_vocab.json` 将汉字映射为数字；同时映射特殊字体，效果如下:

  ```txt
  Loaded vocabulary with 97741 tokens from .\assets\ids_vocab.json
  tongji university 同济大学
  
  Input IDs: [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10, 7550, 30628, 4, 5, 14385, 27468, 4, 16, 6652, 6838, 4, 6, 88848, 10060, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  Token Types: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  Decoded: <BOS><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK><UNK>⿵冂𠮛<SEP>⿰氵齐<SEP>⿻一人<SEP>⿱𰃮子<EOS><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD><PAD>
  ```

* `models/ids_text_embedder.py` 实现维度改变，将 IDS Embedding 经过两级位置编码（文本内部位置；在图像中的位置 x1,y1），然后再将所有文字拼作一个整体

* `utils/ids_data_processor.py` 实现数据的 pipeline 处理，在进入主模块之前，将输入图像、掩码、位置信息、prompt、文字做统一处理



但实际上，现在要等原论文的 training code 发布才能继续了，还是要看看原论文的训练方式，才能在自己的模块上继续训练
