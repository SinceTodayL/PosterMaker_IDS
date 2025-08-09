# PosterMaker IDS训练系统

基于IDS（汉字结构化描述）的中文海报生成模型训练系统。本系统实现了完整的Stage 1训练流水线，支持端到端的IDS文本理解和海报生成。

## 项目概述

PosterMaker是基于Stable Diffusion 3的双ControlNet架构生成模型，包含SceneGenNet和TextRenderNet两个控制网络。本训练系统专注于升级TextRenderNet从静态特征系统到可学习的IDS系统。

### 核心特性

- IDS文本处理：将中文字符转换为结构化序列表示
- 结构化注意力：专门的Transformer理解汉字组合关系  
- Stage 1训练策略：冻结主模型，只训练IDS模块和适配器
- 混合精度训练：支持FP16减少显存占用
- 实时验证：训练过程中自动生成图像监控进度
- 完整推理系统：支持训练后的模型部署使用

### 技术架构

训练系统包含以下核心组件：
- IDSTextEmbedder：IDS序列处理和结构化理解
- LinearAdapter：特征维度适配（128维到4096维）
- 完整训练引擎：混合精度、梯度累积、学习率调度
- 数据加载系统：动态文本编辑任务生成
- 视觉验证系统：实时diffusion推理生成验证图像

## 快速开始

### 环境安装

安装Python依赖包：
```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch >= 2.0.0
- diffusers >= 0.29.2  
- transformers >= 4.43.3
- PIL, numpy, opencv-python
- tqdm, tensorboard, wandb

### 数据集准备

数据集已配置在training_code/dataset目录，结构如下：
```
training_code/dataset/
├── train/
│   ├── 1/
│   │   ├── image.png          # 海报图片
│   │   ├── annotation.json    # 文本标注
│   │   └── subject_mask.png   # 主体掩码（可选）
│   └── 2/...
└── val/
    └── [相同结构]
```

annotation.json文件格式：
```json
{
    "prompt": "海报场景描述",
    "texts": [
        {
            "content": "中文文本内容",
            "pos": [x_min, y_min, x_max, y_max]
        }
    ]
}
```

### 训练执行

验证数据集格式：
```bash
python prepare_data.py --config configs/train_config.yaml
```

开始Stage 1训练：
```bash
python train.py --config configs/train_config.yaml
```

### 模型推理

使用训练完成的模型生成海报：
```bash
python inference.py \
  --config configs/train_config.yaml \
  --checkpoint_path training_output/checkpoints/best_model.pth \
  --text_json_path your_text.json \
  --output_path result.png
```

## 项目结构

```
training_code/
├── configs/
│   └── train_config.yaml      # 训练配置文件
├── src/
│   ├── trainer.py             # 训练引擎核心
│   ├── dataset.py             # 数据加载和处理
│   ├── model_loader.py        # 模型加载和权重管理
│   ├── config.py              # 配置文件加载
│   ├── models/
│   │   ├── ids_text_embedder.py    # IDS文本嵌入器
│   │   ├── adapter_models.py       # 适配器模型
│   │   └── controlnet_sd3.py       # SD3 ControlNet
│   └── utils/
│       ├── ids_query.py        # IDS查询系统
│       ├── ids_tokenizer.py    # IDS分词器
│       └── text_utils.py       # 文本处理工具
├── dataset/                    # 训练数据集目录
├── test_files/                 # 测试脚本目录
├── train.py                    # 主训练脚本
├── inference.py                # 推理脚本
├── prepare_data.py             # 数据验证脚本
└── requirements.txt            # Python依赖
```

## 配置说明

主要配置文件为configs/train_config.yaml，包含以下关键参数：

### 路径配置
- poster_maker_dir: "../PosterMaker" （原始项目路径）
- dataset_dir: "./dataset" （数据集路径）
- output_dir: "./training_output" （训练输出路径）

### 训练超参数
- learning_rate: 1.0e-4 （学习率）
- batch_size: 1 （批次大小）
- gradient_accumulation_steps: 4 （梯度累积步数）
- num_train_epochs: 20 （训练轮数）
- embedding_dim: 64 （IDS嵌入维度）
- max_seq_length: 128 （最大序列长度）
- validation_steps: 100 （验证频率）
- use_amp: True （混合精度训练）

### 其他参数
- lr_scheduler_type: "cosine" （学习率调度器）
- max_grad_norm: 1.0 （梯度裁剪）
- weight_decay: 0.01 （权重衰减）
- seed: 42 （随机种子）

## 训练流程详解

### Stage 1训练策略

Stage 1采用权重冻结策略，只训练必要组件：
- 冻结组件：SD3 VAE、SD3 Transformer、TextRenderNet卷积层
- 可训练组件：IDSTextEmbedder、LinearAdapter
- 训练目标：学习IDS到文本特征的映射关系

### 数据处理流程

1. 动态样本生成：随机选择图像中的文本框进行编辑
2. 文本掩码创建：为选中的文本区域创建conditioning mask
3. IDS分词：将中文文本转换为IDS序列
4. 特征提取：通过IDSTextEmbedder处理IDS序列
5. 适配器转换：将128维特征转换为4096维SD3输入

### 损失函数计算

训练使用标准的diffusion去噪损失：
- 主损失：预测噪声与真实噪声的MSE损失
- 扩展性：预留接口支持感知损失、对抗损失等

### 验证和监控

训练过程包含两种验证方式：
- 数值验证：计算验证集上的损失值
- 视觉验证：生成完整的diffusion推理图像
- 自动保存：保存最佳验证损失对应的模型权重

## 硬件要求

### 最低配置
- GPU显存：10GB（batch_size=1，混合精度）
- 系统内存：16GB
- 存储空间：20GB（包含模型和输出）

### 推荐配置  
- GPU显存：16GB以上
- 系统内存：32GB
- 存储空间：50GB以上

### 性能预期
- 每个epoch训练时间：2-4小时（取决于数据集大小）
- 验证图像生成：每100步约2-3分钟
- 完整Stage 1训练：24-80小时（20个epochs）

## 数据集构建指南

### 数据收集方法

**现有数据集利用**
- 中文OCR数据集（ICDAR、COCO-Text等）
- 商业海报设计素材
- 广告图片数据集

**自动标注工具**
使用OCR工具自动检测文字位置：
```python
import easyocr
reader = easyocr.Reader(['ch_sim', 'en'])
results = reader.readtext(image_path)
```

**手动标注工具**
- LabelImg：免费的边界框标注工具
- CVAT：在线标注平台
- 自定义标注脚本：基于tkinter的简单工具

### 数据质量要求

**图像要求**
- 格式：PNG或JPG，推荐1024x1024
- 内容：包含清晰的中文文字
- 质量：文字边界清晰，无模糊

**标注要求**
- 文本内容：非空的中文字符串
- 边界框：[x_min, y_min, x_max, y_max]格式
- 坐标精度：像素级别，边界框包含完整文字

**数据集规模**
- 最小规模：训练集100个，验证集20个
- 推荐规模：训练集1000个，验证集200个  
- 大规模：训练集10000个以上

## 训练输出说明

### 检查点文件
- best_model.pth：最佳验证损失对应的模型
- checkpoint_epoch_N.pth：每个epoch的检查点
- 只保存可训练部分：IDSTextEmbedder和Adapter的权重

### 验证图像
training_output/validation_samples/目录包含：
- generated_epoch_N_step_M.png：生成的海报图像
- conditioning_epoch_N_step_M.png：输入的conditioning图像
- original_epoch_N_step_M.png：原始参考图像

### 训练日志
- 控制台输出：实时损失值和训练进度
- 文件日志：详细的训练过程记录
- 可选tensorboard/wandb支持

## 推理使用指南

### 文本输入格式

创建文本描述JSON文件：
```json
{
  "prompt": "海报描述",
  "texts": [
    {
      "content": "欢迎光临",
      "pos": [100, 200, 400, 250]
    },
    {
      "content": "新年快乐", 
      "pos": [150, 300, 500, 350]
    }
  ]
}
```

### 推理参数调整
- num_steps：推理步数，默认28，更多步数质量更高
- subject_image：可选的背景图像路径
- output_path：生成图像保存路径

### 批量推理
可以编写脚本批量处理多个文本输入：
```python
import glob
for text_file in glob.glob("texts/*.json"):
    output_name = text_file.replace(".json", "_result.png")
    # 调用推理脚本
```

## 测试和验证工具

系统提供了3个测试脚本，帮助验证环境和组件是否正常工作：

### 快速环境检查
```bash
cd test_files
python quick_test.py
```
验证基本环境：配置加载、模块导入、CUDA设备、基础tensor操作。这是最快的检查方式，适用于：
- 初次搭建环境后的快速验证
- 依赖包安装后的健康检查
- 基本功能是否正常

### 模型加载测试
```bash
cd test_files  
python test_model_loader.py --config ../configs/train_config.yaml
```
深度测试模型加载流程：
- 验证所有预训练模型能否正确加载
- 检查权重冻结策略是否生效  
- 统计可训练参数数量
- 验证模型结构完整性

**使用场景**：
- 修改模型结构后验证
- 更换预训练权重后测试
- 调试模型加载问题

### 完整训练流程测试
```bash
cd test_files
python test_training_pipeline.py --config ../configs/train_config.yaml
```
端到端集成测试：
- 创建虚拟数据集进行训练测试
- 验证完整的前向传播和反向传播
- 测试检查点保存和加载
- 验证视觉验证功能

**使用场景**：
- 正式训练前的最终验证
- 调试训练流程问题
- 验证新功能集成

### 测试建议使用顺序
1. **quick_test.py** - 环境基础检查
2. **test_model_loader.py** - 模型加载验证  
3. **test_training_pipeline.py** - 完整流程测试
4. 开始正式训练

## 常见问题解决

### 训练相关问题

**显存不足错误**
- 减小batch_size到1
- 启用混合精度：use_amp: true
- 减少max_seq_length

**数据加载错误**  
- 检查数据集路径配置
- 运行prepare_data.py验证数据格式
- 确认annotation.json格式正确

**收敛问题**
- 检查学习率设置（推荐1e-4）
- 确认梯度累积步数设置
- 观察验证图像质量变化

### 推理相关问题

**模型加载失败**
- 确认checkpoint_path指向正确文件
- 检查配置文件中的模型路径
- 验证预训练权重文件存在

**生成质量问题**
- 增加推理步数（--num_steps 50）
- 调整文本位置坐标
- 检查输入文本的IDS分词结果

### 环境相关问题

**依赖安装失败**
- 使用conda环境管理依赖
- 分别安装torch和diffusers
- 检查CUDA版本兼容性

**路径问题**
- 使用绝对路径避免相对路径错误
- 检查Windows/Linux路径分隔符
- 确认所有必要文件存在

## 系统扩展

### Stage 2训练准备
完成Stage 1后可考虑Stage 2：
- 解冻SceneGenNet进行端到端微调
- 使用更低学习率（1e-5）
- 添加感知损失和对抗损失

### 模型优化
- LoRA适配：减少可训练参数量
- 量化部署：INT8量化减少推理开销
- 推理加速：使用TensorRT等推理引擎

### 功能扩展
- 多语言支持：扩展到其他语言的文字渲染
- 风格控制：添加字体、颜色等控制参数
- 交互界面：开发Web或桌面应用界面

本系统提供完整的IDS-based文本渲染训练解决方案，支持从数据准备到模型部署的全流程。