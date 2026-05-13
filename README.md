# EANF-Net

本项目是一个基于注意力机制和多层感知机融合的深度学习模型（Enhanced Attention MLP Fusion Network），用于生物医学文本关系抽取任务。项目支持多个基准数据集和多种预训练语言模型。

---

## 目录结构

```
EANF_Net/
├── data/                        # 数据集文件夹
│   ├── ChemProtMS.zip           # ChemProtMS 数据集（train/dev/test）
│   ├── bioinfer.zip             # BioInfer 数据集（train/dev/test）
│   └── ddi2013ms.zip            # DDI2013MS 数据集（train/dev/test）
│
└── original_file/               # 源代码与可执行脚本
    ├── CreateModel.py           # 模型构建代码（各模型定义）
    ├── EnhancedAttentionMLPFusion_ChemProtMS.py  # EANF-Net 在 ChemProtMS 数据集的执行代码
    ├── EnhancedAttentionMLPFusion_bioinfer.py    # EANF-Net 在 BioInfer 数据集的执行代码
    ├── EnhancedAttentionMLPFusion_ddi2013ms.py     # EANF-Net 在 DDI2013MS 数据集的执行代码
    ├── SARE_ChemProtMS.py       # SARE 模型在 ChemProtMS 数据集的执行代码
    ├── SARE_bioinfer.py         # SARE 模型在 BioInfer 数据集的执行代码
    ├── SARE_ddi.py              # SARE 模型在 DDI2013MS 数据集的执行代码
    ├── SaveCsvData.py           # 保存训练结果数据（CSV格式）
    ├── SavePicData.py           # 生成并保存训练过程图表（plt图形）
    ├── base_bioinfer.py         # 基线模型在 BioInfer 数据集的执行代码
    ├── base_chem.py             # 基线模型在 ChemProtMS 数据集的执行代码
    └── base_ddi.py              # 基线模型在 DDI2013MS 数据集的执行代码
```

---

## 文件用途说明

### 1. data/ — 数据集文件夹

存放三个生物医学关系抽取基准数据集，每个数据集均包含 **训练集（train）、验证集（dev）、测试集（test）**：

| 文件 | 说明 |
|------|------|
| `ChemProtMS.zip` | ChemProtMS 数据集 — 化学蛋白质关系抽取 |
| `bioinfer.zip` | BioInfer 数据集 — 生物实体关系抽取 |
| `ddi2013ms.zip` | DDI2013MS 数据集 — 药物-药物相互作用关系抽取 |

> 使用前请将 `.zip` 文件解压至对应目录。

---

### 2. original_file/ — 源代码与可执行脚本

#### 模型定义

| 文件 | 说明 |
|------|------|
| `CreateModel.py` | **模型构建代码**，定义了本项目所使用的各个模型结构（如 EANF-Net、SARE 等），包含网络层搭建、注意力模块、MLP 融合模块等核心组件。 |

#### EANF-Net 执行代码（Enhanced Attention MLP Fusion）

| 文件 | 说明 |
|------|------|
| `EnhancedAttentionMLPFusion_ChemProtMS.py` | EANF-Net 模型在 **ChemProtMS** 数据集上的完整训练、验证与测试脚本 |
| `EnhancedAttentionMLPFusion_bioinfer.py` | EANF-Net 模型在 **BioInfer** 数据集上的完整训练、验证与测试脚本 |
| `EnhancedAttentionMLPFusion_ddi2013ms.py` | EANF-Net 模型在 **DDI2013MS** 数据集上的完整训练、验证与测试脚本 |

#### SARE 模型执行代码

| 文件 | 说明 |
|------|------|
| `SARE_ChemProtMS.py` | SARE 模型在 **ChemProtMS** 数据集上的训练与评估脚本 |
| `SARE_bioinfer.py` | SARE 模型在 **BioInfer** 数据集上的训练与评估脚本 |
| `SARE_ddi.py` | SARE 模型在 **DDI2013MS** 数据集上的训练与评估脚本 |

#### 基线模型执行代码（PubMedBERT / SciBERT / BlueBERT / BioBERT）

| 文件 | 说明 |
|------|------|
| `base_chem.py` | 四种预训练语言模型（PubMedBERT、SciBERT、BlueBERT、BioBERT）在 **ChemProtMS** 数据集上的基准实验脚本 |
| `base_bioinfer.py` | 四种预训练语言模型在 **BioInfer** 数据集上的基准实验脚本 |
| `base_ddi.py` | 四种预训练语言模型在 **DDI2013MS** 数据集上的基准实验脚本 |

#### 结果保存与可视化

| 文件 | 说明 |
|------|------|
| `SaveCsvData.py` | **保存训练结果数据**，将训练过程中的指标（如 loss、accuracy、F1-score 等）导出为 CSV 格式，便于后续分析与对比 |
| `SavePicData.py` | **生成 plt 图形**，绘制训练曲线（如损失曲线、准确率曲线、F1 曲线等）并保存为图片文件 |

---

## 快速开始

1. **准备数据**：解压 `data/` 目录下的各数据集压缩包。
2. **运行基线实验**：
   ```bash
   python original_file/base_chem.py
   ```
3. **运行 EANF-Net**：
   ```bash
   python original_file/EnhancedAttentionMLPFusion_ChemProtMS.py
   ```
4. **保存结果**：训练完成后，使用 `SaveCsvData.py` 和 `SavePicData.py` 导出数据与图表。

---

## 环境依赖

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy / Pandas
- Matplotlib

---

## 数据集说明

| 数据集 | 领域 | 关系类型数 | 说明 |
|--------|------|-----------|------|
| ChemProtMS | 化学-蛋白质 | 多类 | 从 PubMed 文献中提取的化学与蛋白质关系 |
| BioInfer | 生物实体 | 多类 | 生物医学文献中的实体关系 |
| DDI2013MS | 药物相互作用 | 多类 | 药物-药物相互作用检测 |

---

## 支持的预训练模型

- **PubMedBERT**：在 PubMed 摘要上预训练的 BERT 模型
- **SciBERT**：在科学文献上预训练的 BERT 模型
- **BlueBERT**：在 PubMed 和 MIMIC-III 上预训练的 BERT 模型
- **BioBERT**：在生物医学文献上预训练的 BERT 模型
