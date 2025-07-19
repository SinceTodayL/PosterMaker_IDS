主干代码来源：https://github.com/alimama-creative/PosterMaker



# PROCESS



>Before 202507

跑通模型，修改部分代码，不影响任何执行逻辑和模型结构，减少运行时的峰值显存



> 20250711

准备添加 IDS 模块，改进 TextrenderNet 

工作：

* 数据集？模型需要学习怎么将汉字映射到 IDS 结构
* 如何训练？训练代码？冻结哪些参数？哪些参数是可训练的？
* 评估方式？包括 损失函数、评估对比方式？

> 论文核心论点 (Thesis Statement): *“相较于现有的字符级（Character-level）文本渲染方法，我们提出了一种基于表意文字描述序列（IDS）的组件级（Component-level）特征组合网络。该方法不仅显著提升了对结构复杂汉字（unseen complex characters）的零样本生成能力和视觉一致性，还能在保持甚至超越原有场景融合能力的同时，为文本的风格化控制提供了新的范式。”*
>
> 基于此，我们的工作需要证明以下几点：
>
> * 优越性: IDS方法在文本渲染上显著优于字符级方法。
>
> * 兼容性: 我们的改进可以无缝集成到现有的大型文生图模型（如PosterMaker）中，不损害其原有能力。
>
> * 扩展性 (Novelty): 我们的方法为未来更精细的文本风格控制（如部首级风格迁移）打开了大门。



>20250712

Step1: 将输入文字查表，转化成标准 IDS 结构

Step2: 用 IDS 结构训练模型， TextRenderNet, Adapter (问题：优化目标是什么？是整个模型一块训练，还是将 TextRenderNet 单独拿出来训练？)

* * 我们想要的通过 IDS 渲染出复杂汉字的方式，是不是一个单独的模型？



>20250719

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





