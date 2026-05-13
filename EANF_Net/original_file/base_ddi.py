import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import csv
from datetime import datetime


############################################# 单模型预测模块 ##############################################

class SingleBERTModel(nn.Module):
    """
    单个BERT模型预测器
    """

    def __init__(self, model_path, model_name, num_labels=5, dropout_rate=0.3):
        super(SingleBERTModel, self).__init__()

        # 存储模型名称
        self.model_name = model_name

        # BERT模型
        self.bert = BertModel.from_pretrained(model_path)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, num_labels)
        )


    def forward(self, input_ids, attention_mask, labels=None):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # 取[CLS] token的输出

        # 分类
        logits = self.classifier(pooled_output)

        # 计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return None, logits


def evaluate_single_model(model, data_loader, device, tokenizer_name):
    """
    评估单个模型 - 简化版本
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_losses = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"评估{tokenizer_name}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            loss, logits = model(input_ids, attention_mask, labels)

            # 收集结果
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

            if loss is not None:
                all_losses.append(loss.item())

    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

    # 计算平均损失
    avg_loss = np.mean(all_losses) if all_losses else 0

    return {
        'model_name': tokenizer_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss
    }


def train_single_model(model, train_loader, dev_loader, epochs=25,
                       learning_rate=2e-5, device='cuda', save_dir='models'):
    """
    训练单个模型 - 仅训练和验证，不测试
    """
    model.to(device)
    model.train()

    # 创建保存目录
    model_save_dir = os.path.join(save_dir, model.model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(model_save_dir, 'training_log.csv')
    best_model_path = os.path.join(model_save_dir, 'best_model.pth')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    print(f"\n开始训练 {model.model_name} 模型...")
    print(f"训练日志将保存到: {log_file}")

    # 创建训练日志CSV文件
    with open(log_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'timestamp'])

    best_f1 = 0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ========== 训练阶段 ==========
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []

        progress_bar = tqdm(train_loader, desc=f"训练 {model.model_name}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            train_predictions.extend(predictions.cpu().tolist())
            train_targets.extend(labels.cpu().tolist())
            train_losses.append(loss.item())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算训练指标
        train_accuracy = accuracy_score(train_targets, train_predictions)
        train_precision = precision_score(train_targets, train_predictions, average='weighted', zero_division=0)
        train_recall = recall_score(train_targets, train_predictions, average='weighted', zero_division=0)
        train_f1 = f1_score(train_targets, train_predictions, average='weighted', zero_division=0)
        avg_train_loss = np.mean(train_losses)

        # ========== 验证阶段 ==========
        model.eval()
        dev_results = evaluate_single_model(model, dev_loader, device, model.model_name)
        dev_accuracy = dev_results['accuracy']
        dev_precision = dev_results['precision']
        dev_recall = dev_results['recall']
        dev_f1 = dev_results['f1_score']
        avg_dev_loss = dev_results['loss']

        # 记录到CSV文件（只记录train和dev）
        with open(log_file, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, 'train', avg_train_loss, train_accuracy, train_precision, train_recall, train_f1,
                 timestamp])
            writer.writerow(
                [epoch + 1, 'dev', avg_dev_loss, dev_accuracy, dev_precision, dev_recall, dev_f1, timestamp])

        # 打印结果
        print(f"训练集 - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"验证集 - Loss: {avg_dev_loss:.4f}, Accuracy: {dev_accuracy:.4f}, F1: {dev_f1:.4f}")

        # 保存最佳模型（基于验证集F1）
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_epoch': best_epoch,
                'train_metrics': {
                    'loss': avg_train_loss,
                    'accuracy': train_accuracy,
                    'f1': train_f1
                },
                'dev_metrics': {
                    'loss': avg_dev_loss,
                    'accuracy': dev_accuracy,
                    'f1': dev_f1
                }
            }, best_model_path)

            print(f"🔥 保存最佳模型 (epoch {best_epoch})，验证集F1: {best_f1:.4f}")

    return best_model_path  # 返回最佳模型路径，供最终测试使用


def save_results_summary(results_dict, save_path='single_models_results_summary.csv'):
    """
    保存四个模型的结果汇总
    """
    # 准备数据
    data = []
    for model_name, result in results_dict.items():
        row = {
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1_score'],
            'Loss': result['loss']
        }
        data.append(row)

    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    # 计算平均值
    avg_row = {
        'Model': 'Average',
        'Accuracy': df['Accuracy'].mean(),
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1_Score': df['F1_Score'].mean(),
        'Loss': df['Loss'].mean()
    }

    # 添加到DataFrame
    df_avg = pd.DataFrame([avg_row])
    df_with_avg = pd.concat([df, df_avg], ignore_index=True)
    df_with_avg.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"结果汇总已保存到: {save_path}")
    print(f"平均F1-Score: {avg_row['F1_Score']:.4f}")


############################################# 主执行函数 ##############################################
def run_single_models_evaluation(train_loaders, dev_loaders, test_loaders,
                                 model_paths, device='cuda',
                                 train_single=False, epochs=25):
    model_configs = {
        'SciBERT': model_paths['scibert'],
        'BioBERT': model_paths['biobert'],
        'BlueBERT': model_paths['bluebert'],
        'PubMedBERT': model_paths['pubmedbert']
    }

    results = {}

    for model_name, model_path in model_configs.items():
        print(f"\n{'=' * 80}")
        print(f" 处理 {model_name} 模型 ")
        print(f"{'=' * 80}")

        train_loader = train_loaders[model_name]
        dev_loader = dev_loaders[model_name]
        test_loader = test_loaders[model_name]

        # 创建模型
        model = SingleBERTModel(
            model_path=model_path,
            model_name=model_name,
            num_labels=5,
            dropout_rate=0.3
        )

        if train_single:
            # 训练模型（过程中不测试）
            best_model_path = train_single_model(
                model=model,
                train_loader=train_loader,
                dev_loader=dev_loader,
                epochs=epochs,
                learning_rate=2e-5,
                device=device,
                save_dir='baseline_ddi_models'
            )
            # 加载最佳模型权重
            ckpt = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"已加载最佳模型，验证集F1: {ckpt['best_f1']:.4f}，epoch: {ckpt['best_epoch']}")

        # 训练结束后，使用测试集进行最终评估
        result = evaluate_single_model(
            model=model,
            data_loader=test_loader,
            device=device,
            tokenizer_name=model_name
        )
        results[model_name] = result

        print(f"\n{model_name} 测试结果:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        print(f"  Loss:      {result['loss']:.4f}")

    save_results_summary(results)
    return results


############################################# 使用示例 ##############################################
# # PubMedBERT
# pubmed_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# pubmed_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# # BioBERT
# bio_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# bio_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# # BlueBERT
# blue_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
# blue_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

# # SciBERT
# sci_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
# sci_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model_paths = {
        'scibert': '',
        'biobert': '',
        'bluebert': '',
        'pubmedbert': ''
    }

    max_len = 300
    batch_sz = 8

    # 一次性把四个模型的 loader 都建好
    from transformers import AutoTokenizer

    tok_scibert = AutoTokenizer.from_pretrained(model_paths['scibert'])
    tok_biobert = AutoTokenizer.from_pretrained(model_paths['biobert'])
    tok_bluebert = AutoTokenizer.from_pretrained(model_paths['bluebert'])
    tok_pubmedbert = AutoTokenizer.from_pretrained(model_paths['pubmedbert'])

    df_train = pd.read_csv('../data/ddi2013ms/train.tsv', sep='\t')
    df_dev = pd.read_csv('../data/ddi2013ms/dev.tsv', sep='\t')
    df_test = pd.read_csv('../data/ddi2013ms/test.tsv', sep='\t')

    label_map = {
        'DDI-false': 0,
        'DDI-effect': 1,
        'DDI-mechanism': 2,
        'DDI-advise': 3,
        'DDI-int': 4
    }

    from torch.utils.data import Dataset, DataLoader


    class SingleModelDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length):
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sentence = self.data['sentence'][idx]
            label_str = self.data['label'][idx]
            label = label_map[label_str]

            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


    def build_loader(df, tok):
        ds = SingleModelDataset(df, tok, max_len)
        return DataLoader(ds, batch_size=batch_sz, shuffle=True)


    #四套 loader 字典
    train_loaders = {
        'SciBERT': build_loader(df_train, tok_scibert),
        'BioBERT': build_loader(df_train, tok_biobert),
        'BlueBERT': build_loader(df_train, tok_bluebert),
        'PubMedBERT': build_loader(df_train, tok_pubmedbert)
    }
    dev_loaders = {
        'SciBERT': build_loader(df_dev, tok_scibert),
        'BioBERT': build_loader(df_dev, tok_biobert),
        'BlueBERT': build_loader(df_dev, tok_bluebert),
        'PubMedBERT': build_loader(df_dev, tok_pubmedbert)
    }
    test_loaders = {
        'SciBERT': build_loader(df_test, tok_scibert),
        'BioBERT': build_loader(df_test, tok_biobert),
        'BlueBERT': build_loader(df_test, tok_bluebert),
        'PubMedBERT': build_loader(df_test, tok_pubmedbert)
    }

    print("=" * 80)
    print("开始四模型独立训练与评估")
    print("=" * 80)

    results = run_single_models_evaluation(
        train_loaders=train_loaders,
        dev_loaders=dev_loaders,
        test_loaders=test_loaders,
        model_paths=model_paths,
        device=device,
        train_single=True,
        epochs=25
    )

    print("\n" + "=" * 80)
    print("四模型独立结果汇总")
    for name, res in results.items():
        print(f"{name:12} - F1: {res['f1_score']:.4f}, Acc: {res['accuracy']:.4f}")