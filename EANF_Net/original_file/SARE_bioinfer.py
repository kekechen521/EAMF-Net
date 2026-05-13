import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#############################################定义 BERT 模型和 tokenizer##############################################
# # BioBERT
# bio_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
# bio_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# # BlueBERT
# blue_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
# blue_model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

# # PubMedBERT
# pubmed_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# pubmed_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

model_path_1 = ''
pubmedbert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
pubmedbert_model = AutoModel.from_pretrained(model_path_1)
model_path_2 = ''
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path_2)
biobert_model = AutoModel.from_pretrained(model_path_2)
model_path_3 = ''
bluebert_tokenizer = AutoTokenizer.from_pretrained(model_path_3)
bluebert_model = AutoModel.from_pretrained(model_path_3)
print("model load")
# ############################################读取数据#################################################################
df_train = pd.read_csv('../data/bioinfer/BioInfer_85train_new.csv')
df_dev = pd.read_csv('../data/bioinfer/BioInfer_dev_new.csv')
df_test = pd.read_csv('../data/bioinfer/BioInfer_test_new.csv')

#######################################################定义模型参数#########################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
num_label=2


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
# {'CPR:3', 'CPR:9', 'CPR:5', 'false', 'CPR:6', 'CPR:4'}
# label_map = {
#     'CPR:3': 0,
#     'CPR:9': 1,
#     'CPR:5': 2,
#     'CPR:6': 3,
#     'CPR:4': 4,
#     'false': 5
#     # 可以根据你的实际标签情况添加更多映射关系
# }

# 定义数据集类
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer_1, tokenizer_2, tokenizer_3, max_length):
        self.data = dataframe
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['sentence'][idx]
        label_str = self.data['label'][idx]
        label = int(label_str)

        encoding_1 = self.tokenizer_1(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding_2 = self.tokenizer_2(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding_3 = self.tokenizer_3(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids_1': encoding_1['input_ids'].flatten(),
            'attention_mask_1': encoding_1['attention_mask'].flatten(),
            'input_ids_2': encoding_2['input_ids'].flatten(),
            'attention_mask_2': encoding_2['attention_mask'].flatten(),
            'input_ids_3': encoding_3['input_ids'].flatten(),
            'attention_mask_3': encoding_3['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义数据加载器
def create_data_loader(df, tokenizer_1, tokenizer_2, tokenizer_3, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

# 定义模型参数
max_length = 300
batch_size = 8

# 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, pubmedbert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)
dev_data_loader = create_data_loader(df_dev, pubmedbert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, pubmedbert_tokenizer,biobert_tokenizer,bluebert_tokenizer, max_length, batch_size)


# #输出data_loader
# for batch in train_data_loader:
#     print(batch)


#####################################################定义模型####################################################


# best model
class YModel(nn.Module):
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5):
        super(YModel, self).__init__()
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)



        self.fc_1 = nn.Linear(768, 128)
        self.fc_2 = nn.Linear(768, 128)
        self.fc_3 = nn.Linear(768, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=1)
        self.fc = nn.Linear(128 * 3, num_labels)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3, labels=None):
        output_1 = self.Bert_1(input_ids=input_ids_1, attention_mask=attention_mask_1)[0]  # output_1 shape: (batch_size, sequence_length, hidden_size)
        output_2 = self.Bert_2(input_ids=input_ids_2, attention_mask=attention_mask_2)[0]  # output_2 shape: (batch_size, sequence_length, hidden_size)
        output_3 = self.Bert_3(input_ids=input_ids_3, attention_mask=attention_mask_3)[0]  # output_3 shape: (batch_size, sequence_length, hidden_size)

        # Apply linear transformation
        output_1 = torch.tanh(self.fc_1(output_1[:, 0]))  # output_1 shape: (batch_size, 128)
        output_2 = torch.tanh(self.fc_2(output_2[:, 0]))  # output_2 shape: (batch_size, 128)
        output_3 = torch.tanh(self.fc_3(output_3[:, 0]))  # output_3 shape: (batch_size, 128)

        # Concatenate the outputs
        combined_output = torch.cat((output_1, output_2, output_3), dim=1)  # combined_output shape: (batch_size, 128 * 3)

        # Apply fully connected layer
        logits = self.fc(combined_output)  # logits shape: (batch_size, num_labels)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return loss,logits
        else:
            return None,logits
        


        

import torch
import torch.nn as nn
from transformers import BertModel
import csv


# 实例化模型
num_labels = 2
model = YModel(model_path_1,model_path_2,model_path_3, num_labels)
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 训练模型
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):

    model.train()
    losses = []
    targets = []
    predictions = []

    for batch in data_loader:
        input_ids_1 = batch['input_ids_1'].to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        input_ids_2 = batch['input_ids_2'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)
        input_ids_3 = batch['input_ids_3'].to(device)
        attention_mask_3 = batch['attention_mask_3'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        loss, logits = model(input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2,
                             input_ids_3=input_ids_3,attention_mask_3=attention_mask_3, labels=labels)


        predictions.extend(torch.argmax(logits, dim=1).tolist())
        targets.extend(labels.tolist())

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if scheduler:
        scheduler.step()

    return losses, targets, predictions

# 验证模型
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    targets = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            input_ids_3 = batch['input_ids_3'].to(device)
            attention_mask_3 = batch['attention_mask_3'].to(device)
            labels = batch['labels'].to(device)

            loss, logits = model(input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,input_ids_2=input_ids_2,attention_mask_2=attention_mask_2,
                             input_ids_3=input_ids_3,attention_mask_3=attention_mask_3, labels=labels)
            
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            targets.extend(labels.tolist())

            losses.append(loss.item())

    return losses, targets, predictions

# 用于存储每个 epoch 的训练和验证指标（同一行）
epoch_records = []

# 训练模型
epochs = 25
best_accuracy = 0
print("start training...")

for epoch in range(epochs):
    # 训练阶段
    train_losses, train_targets, train_predictions = train_epoch(model, train_data_loader, loss_fn, optimizer, device)
    train_loss = sum(train_losses) / len(train_losses)
    train_accuracy = accuracy_score(train_targets, train_predictions)
    train_precision = precision_score(train_targets, train_predictions, average='weighted', zero_division=0)
    train_recall = recall_score(train_targets, train_predictions, average='weighted', zero_division=0)
    train_f1 = f1_score(train_targets, train_predictions, average='weighted', zero_division=0)

    # 验证阶段
    dev_losses, dev_targets, dev_predictions = eval_model(model, dev_data_loader, loss_fn, device)
    dev_loss = sum(dev_losses) / len(dev_losses)
    dev_accuracy = accuracy_score(dev_targets, dev_predictions)
    dev_precision = precision_score(dev_targets, dev_predictions, average='weighted', zero_division=0)
    dev_recall = recall_score(dev_targets, dev_predictions, average='weighted', zero_division=0)
    dev_f1 = f1_score(dev_targets, dev_predictions, average='weighted', zero_division=0)

    # 保存到内存记录（同一行）
    epoch_records.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'dev_loss': dev_loss,
        'dev_accuracy': dev_accuracy,
        'dev_precision': dev_precision,
        'dev_recall': dev_recall,
        'dev_f1': dev_f1
    })

    # 打印信息
    print(f'Epoch {epoch + 1}/{epochs}')
    print(
        f'  Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}')
    print(
        f'  Dev Loss:   {dev_loss:.4f}, Acc: {dev_accuracy:.4f}, Prec: {dev_precision:.4f}, Rec: {dev_recall:.4f}, F1: {dev_f1:.4f}')

    # 保存最佳模型（基于验证准确率）
    if dev_f1 > best_accuracy:
        best_accuracy = dev_f1
        torch.save(model.state_dict(), 'sare_model_bioInfer.pth')
        print(f'  🔥 Best model saved (Dev Acc: {best_accuracy:.4f})')

print("Training completed. Saving all epoch records to CSV...")

# 训练结束后统一保存到 CSV（每个 epoch 一行）
csv_filename = 'training_log_bioinfer.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头（训练和验证指标合并在一行）
    writer.writerow([
        'epoch',
        'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
        'dev_loss', 'dev_accuracy', 'dev_precision', 'dev_recall', 'dev_f1'
    ])
    for record in epoch_records:
        writer.writerow([
            record['epoch'],
            record['train_loss'], record['train_accuracy'], record['train_precision'],
            record['train_recall'], record['train_f1'],
            record['dev_loss'], record['dev_accuracy'], record['dev_precision'],
            record['dev_recall'], record['dev_f1']
        ])

print(f"Training log saved to {csv_filename}")

# 加载最佳模型进行测试
model.load_state_dict(torch.load('sare_model_bioInfer.pth'))
model.to(device)

test_losses, test_targets, test_predictions = eval_model(model, test_data_loader, loss_fn, device)
test_loss = sum(test_losses) / len(test_losses)
test_accuracy = accuracy_score(test_targets, test_predictions)
test_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)

# 将测试结果追加到 CSV（单独一行，用 test 标记 epoch 列）
with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'test',
        test_loss, test_accuracy, test_precision, test_recall, test_f1,
        '', '', '', '', ''  # 验证集相关列留空，保持列数一致
    ])

print("\nFinal Test Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F1 Score: {test_f1:.4f}")
print(f"\nAll results saved to {csv_filename}")