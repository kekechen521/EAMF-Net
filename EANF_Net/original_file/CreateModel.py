import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import torch.nn.functional as F

#EAMF-Net
class EAMF_Net(nn.Module):
    """
    自注意力 + 交叉注意力 + MLP特征融合 + 多层分类器
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }
        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)
        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        # 交叉注意力层
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        # MLP特征融合
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, hidden_dim * 3),
        )
        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" EAMF-Net 模型配置")
        print("  有多层分类器，交叉注意力机制，MLP特征融合，没有MLP特征权重融合")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：多层分类器")
        print(f"   • 注意力机制: 自注意力 + 三重交叉注意力")
        print(f"   • 融合方式: MLP深度融合 + 残差连接")
        print(f"   • 使用自注意力与交叉注意力的特征平均融合替代MLP注意力加权特征融合")
        print("=" * 60 + "\n")


    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 交叉注意力
        cross_attended_1, cross_weights_1 = self.cross_attention_1(
            feat1.unsqueeze(1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_2, cross_weights_2 = self.cross_attention_2(
            feat2.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_3, cross_weights_3 = self.cross_attention_3(
            feat3.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        )


        # 3. 特征平均融合 (代替权重融合)
        attended_feat1 = (feat1 + cross_attended_1.squeeze(1)) / 2
        attended_feat2 = (feat2 + cross_attended_2.squeeze(1)) / 2
        attended_feat3 = (feat3 + cross_attended_3.squeeze(1)) / 2

        # 4. 拼接特征
        concat_features = torch.cat([attended_feat1, attended_feat2, attended_feat3], dim=1)

        # 5. MLP特征融合
        mlp_fused = self.mlp_fusion(concat_features)
        mlp_fused = self.layer_norm2(mlp_fused + concat_features)
        mlp_fused = self.dropout(F.gelu(mlp_fused))
        # 6. 多层分类
        logits = self.classifier(mlp_fused)
        #返回模型权重
        if return_weights:
            # 提取交叉注意力权重
            cross_weight_1 = cross_weights_1.mean(dim=1).squeeze(1)
            cross_weight_2 = cross_weights_2.mean(dim=1).squeeze(1)
            cross_weight_3 = cross_weights_3.mean(dim=1).squeeze(1)
            cross_weights = torch.cat([cross_weight_1, cross_weight_2, cross_weight_3], dim=1)  # [batch, 3]


        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, cross_weights
            return loss, logits
        else:
            if return_weights:
                return None, logits, cross_weights
            return None, logits

#Baseline_model 3个基础Bert模型+自注意力机制+单层分类器
class Baseline_model(nn.Module):
    """
    消融实验: 3个基础Bert + 自注意力 + 单层分类器
    移除MLP特征融合、交叉注意力
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 单层分类器
        self.classifier = nn.Linear(hidden_dim * 3, num_labels)

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" Baseline_model 模型配置")
        print(" 消融实验: 有单层分类器，没有MLP特征融合，交叉注意力机制,多层分类器")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：单层分类器")
        print(f"   • 注意力机制: 自注意力 ")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 特征拼接
        concat_features = torch.cat([feat1, feat2, feat3], dim=1)
        concat_features = self.dropout(F.gelu(concat_features))

        # 3. 单层分类
        logits = self.classifier(concat_features)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, None
            return loss, logits
        else:
            if return_weights:
                return None, logits, None
            return None, logits

#Baseline-Multi 3个基础Bert模型+自注意力机制+多层分类器
class Baseline_Multi(nn.Module):
    """消融实验: 只有多层分类器，无交叉注意力、MLP融合"""

    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=16, dropout_rate=0.3):
        super(Baseline_Multi, self).__init__()

        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" Baseline_Multi 模型配置")
        print(" 消融实验: 有多层分类器，没有MLP特征融合，交叉注意力机制")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：多层分类器")
        print(f"   • 注意力机制: 自注意力")
        print(f"   • 融合方式: 注意力特征简单拼接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2.特征拼接
        concat_features = torch.cat([feat1, feat2, feat3], dim=1)
        concat_features = self.dropout(F.gelu(concat_features))

        # 3. 直接分类
        logits = self.classifier(concat_features)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                # 无权重融合，返回自注意力权重
                return loss, logits, None
            return loss, logits
        else:
            if return_weights:
                return None, logits, None
            return None, logits

#Baseline_CA 3个基础Bert模型+自注意力机制+交叉注意力机制+单层分类器
class Baseline_CA(nn.Module):
    """
    消融实验: 3个基础Bert + 自注意力 + 交叉注意力 + 单层分类器
    移除MLP特征融合、多层分类器
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 交叉注意力层
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 单层分类器
        self.classifier = nn.Linear(hidden_dim * 3, num_labels)

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" Baseline-CA 模型配置")
        print(" 消融实验: 有单层分类器，交叉注意力机制，没有MLP特征融合，多层分类器")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：单层分类器")
        print(f"   • 注意力机制: 自注意力 + 三重交叉注意力")
        print(f"   • 融合方式: 自注意力与交叉注意力特征平均拼接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 交叉注意力
        cross_attended_1, cross_weights_1 = self.cross_attention_1(
            feat1.unsqueeze(1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_2, cross_weights_2 = self.cross_attention_2(
            feat2.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_3, cross_weights_3 = self.cross_attention_3(
            feat3.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        )

        # 3. 特征平均融合 (代替权重融合)
        attended_feat1 = (feat1 + cross_attended_1.squeeze(1)) / 2
        attended_feat2 = (feat2 + cross_attended_2.squeeze(1)) / 2
        attended_feat3 = (feat3 + cross_attended_3.squeeze(1)) / 2


        # 4. 特征拼接
        concat_features = torch.cat([attended_feat1, attended_feat2, attended_feat3], dim=1)
        concat_features = self.dropout(F.gelu(concat_features))

        # 5. 单层分类
        logits = self.classifier(concat_features)

        if return_weights:
            # 提取交叉注意力权重
            cross_weight_1 = cross_weights_1.mean(dim=1).squeeze(1)
            cross_weight_2 = cross_weights_2.mean(dim=1).squeeze(1)
            cross_weight_3 = cross_weights_3.mean(dim=1).squeeze(1)
            cross_weights = torch.cat([cross_weight_1, cross_weight_2, cross_weight_3], dim=1)  # [batch, 6]

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, cross_weights
            return loss, logits
        else:
            if return_weights:
                return None, logits, cross_weights
            return None, logits

#Baseline_CA_C 3个基础Bert模型+自注意力机制+交叉注意力机制+单层分类器+6个特征直接拼接
class Baseline_CA_C(nn.Module):
    """
    消融实验: 3个基础Bert + 自注意力 + 交叉注意力 + 单层分类器+6个特征直接拼接
    移除MLP特征融合
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 交叉注意力层
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 单层分类器
        self.classifier = nn.Linear(hidden_dim * 6, num_labels)

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" Baseline_CA_C 模型配置")
        print(" 消融实验: 有单层分类器，交叉注意力机制,6个特征直接拼接，没有MLP特征融合，多层分类器")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：单层分类器")
        print(f"   • 注意力机制: 自注意力 + 三重交叉注意力")
        print(f"   • 融合方式: 自注意力与交叉注意力特征直接拼接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))


        # 2. 交叉注意力
        cross_attended_1, cross_weights_1 = self.cross_attention_1(
            feat1.unsqueeze(1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_2, cross_weights_2 = self.cross_attention_2(
            feat2.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_3, cross_weights_3 = self.cross_attention_3(
            feat3.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        )

        # 3. 特征拼接6个特征
        concat_features = torch.cat([
            feat1, cross_attended_1.squeeze(1),
            feat2, cross_attended_2.squeeze(1),
            feat3, cross_attended_3.squeeze(1)
        ], dim=1)
        concat_features = self.dropout(F.gelu(concat_features))

        # 4. 单层分类
        logits = self.classifier(concat_features)

        if return_weights:
            # 提取交叉注意力权重
            cross_weight_1 = cross_weights_1.mean(dim=1).squeeze(1)
            cross_weight_2 = cross_weights_2.mean(dim=1).squeeze(1)
            cross_weight_3 = cross_weights_3.mean(dim=1).squeeze(1)
            cross_weights = torch.cat([cross_weight_1, cross_weight_2, cross_weight_3], dim=1)  # [batch, 6]

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, cross_weights
            return loss, logits
        else:
            if return_weights:
                return None, logits, cross_weights
            return None, logits

#Baseline-MLP 3个基础Bert模型+自注意力机制+MLP特征融合+单层分类器
class Baseline_MLP(nn.Module):
    """
    消融实验: 自注意力 + MLP特征融合 + 单层分类器
    移除交叉注意力、MLP权重融合
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=16, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }
        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)
        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)
        # 仅保留自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        # MLP特征融合
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, hidden_dim * 3),
        )
        # 单层分类器
        self.classifier = nn.Linear(hidden_dim * 3, num_labels)
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" Baseline-MLP 模型配置")
        print(" 消融实验: 有单层分类器，MLP特征融合，没有多层分类器，交叉注意力机制")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：单层分类器")
        print(f"   • 注意力机制: 自注意力 ")
        print(f"   • 融合方式: MLP深度融合 + 残差连接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 特征拼接
        concat_features = torch.cat([feat1, feat2, feat3], dim=1)

        # 3. MLP特征融合
        mlp_fused = self.mlp_fusion(concat_features)
        mlp_fused = self.layer_norm2(mlp_fused + concat_features)
        mlp_fused = self.dropout(F.gelu(mlp_fused))

        # 4. 单层分类
        logits = self.classifier(mlp_fused)
        # 损失计算
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, None
            return loss, logits
        else:
            if return_weights:
                return None, logits, None
            return None, logits

#EAMF-Net-withoutCA 3个基础Bert模型+自注意力机制+MLP特征融合+多层分类器
class EAMF_Net_withoutCA(nn.Module):
    """
    消融实验: 自注意力 + MLP特征融合 + 多层分类器
    移除交叉注意力、MLP权重融合
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=16, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }
        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)
        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)
        # 仅保留自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        # MLP特征融合
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, hidden_dim * 3),
        )
        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" EAMF-Net-withoutCA 模型配置")
        print(" 消融实验: 有多层分类器，MLP特征融合，没有交叉注意力机制")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：多层分类器")
        print(f"   • 注意力机制: 自注意力 ")
        print(f"   • 融合方式: MLP深度融合 + 残差连接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 特征拼接
        concat_features = torch.cat([feat1, feat2, feat3], dim=1)

        # 3. MLP特征融合
        mlp_fused = self.mlp_fusion(concat_features)
        mlp_fused = self.layer_norm2(mlp_fused + concat_features)
        mlp_fused = self.dropout(F.gelu(mlp_fused))

        # 4. 多层分类
        logits = self.classifier(mlp_fused)
        # 损失计算
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, None
            return loss, logits
        else:
            if return_weights:
                return None, logits, None
            return None, logits

#EAMF-Net-withoutmlp 3个基础Bert模型+自注意力机制+交叉注意力机制+多层分类器
class EAMF_Net_withoutMLP(nn.Module):
    """
    消融实验: 3个基础Bert + 自注意力 + 交叉注意力 + 多层分类器
    移除MLP特征融合
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 交叉注意力层
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" EAMF-Net-withoutmlp 模型配置")
        print(" 消融实验: 有多层分类器，交叉注意力机制，没有MLP特征融合、MLP特征权重融合")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：多层分类器")
        print(f"   • 注意力机制: 自注意力 + 三重交叉注意力")
        print(f"   • 融合方式: 自注意力与交叉注意力特征平均拼接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))


        # 2. 交叉注意力
        cross_attended_1, cross_weights_1 = self.cross_attention_1(
            feat1.unsqueeze(1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_2, cross_weights_2 = self.cross_attention_2(
            feat2.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_3, cross_weights_3 = self.cross_attention_3(
            feat3.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        )

        # 3. 融合自/交叉注意力特征
        attended_feat1 = (feat1 + cross_attended_1.squeeze(1)) / 2
        attended_feat2 = (feat2 + cross_attended_2.squeeze(1)) / 2
        attended_feat3 = (feat3 + cross_attended_3.squeeze(1)) / 2

        # 4. 特征拼接
        concat_features = torch.cat([attended_feat1, attended_feat2, attended_feat3], dim=1)
        concat_features = self.dropout(F.gelu(concat_features))


        # 6. 多层分类
        logits = self.classifier(concat_features)

        if return_weights:
            # 提取交叉注意力权重
            cross_weight_1 = cross_weights_1.mean(dim=1).squeeze(1)
            cross_weight_2 = cross_weights_2.mean(dim=1).squeeze(1)
            cross_weight_3 = cross_weights_3.mean(dim=1).squeeze(1)
            cross_weights = torch.cat([cross_weight_1, cross_weight_2, cross_weight_3], dim=1)  # [batch, 6]


        # 损失计算
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, cross_weights
            return loss, logits
        else:
            if return_weights:
                return None, logits, cross_weights
            return None, logits

#EAMF-Net-withoutmlp—C 3个基础Bert模型+自注意力机制+交叉注意力机制+多层分类器+6个特征直接拼接
class EAMF_Net_withoutMLP_C(nn.Module):
    """
    消融实验: 3个基础Bert + 自注意力 + 交叉注意力 + 多层分类器 + 6个特征直接拼接
    移除MLP特征融合
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 交叉注意力层
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 多层分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" EAMF_Net_withoutMLP_C 模型配置")
        print(" 消融实验: 有多层分类器，交叉注意力机制，6个特征直接拼接，没有MLP特征融合")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：多层分类器")
        print(f"   • 注意力机制: 自注意力 + 三重交叉注意力")
        print(f"   • 融合方式: 自注意力与交叉注意力特征直接拼接")
        print("=" * 60 + "\n")

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 交叉注意力
        cross_attended_1, cross_weights_1 = self.cross_attention_1(
            feat1.unsqueeze(1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_2, cross_weights_2 = self.cross_attention_2(
            feat2.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_3, cross_weights_3 = self.cross_attention_3(
            feat3.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        )

        # 3. 特征拼接6个特征
        concat_features = torch.cat([
            feat1, cross_attended_1.squeeze(1),
            feat2, cross_attended_2.squeeze(1),
            feat3, cross_attended_3.squeeze(1)
        ], dim=1)
        concat_features = self.dropout(F.gelu(concat_features))

        # 4. 多层分类
        logits = self.classifier(concat_features)

        if return_weights:
            # 提取交叉注意力权重
            cross_weight_1 = cross_weights_1.mean(dim=1).squeeze(1)
            cross_weight_2 = cross_weights_2.mean(dim=1).squeeze(1)
            cross_weight_3 = cross_weights_3.mean(dim=1).squeeze(1)
            cross_weights = torch.cat([cross_weight_1, cross_weight_2, cross_weight_3], dim=1)  # [batch, 6]

        # 损失计算
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, cross_weights
            return loss, logits
        else:
            if return_weights:
                return None, logits, cross_weights
            return None, logits

#EAMF-Net-Single 3个基础Bert模型+自注意力机制+交叉注意力机制+MLP特征融合+单层分类器
class EAMF_Net_Single(nn.Module):
    """
    消融实验: 3个基础Bert + 自注意力 + 交叉注意力 + MLP特征融合 + 单层分类器
    移除多层分类器
    """
    def __init__(self, model_name_1, model_name_2, model_name_3, num_labels=5,
                 hidden_dim=128, mlp_hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = BertModel.from_pretrained(model_name_1)
        self.Bert_2 = BertModel.from_pretrained(model_name_2)
        self.Bert_3 = BertModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # 交叉注意力层
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.cross_attention_3 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # MLP特征融合
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, hidden_dim * 3),
        )

        # 单层分类器
        self.classifier = nn.Linear(hidden_dim * 3, num_labels)

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)

        # 打印模型参数信息
        self._print_model_config()

    def _print_model_config(self):
        """打印模型配置参数"""
        print("\n" + "=" * 60)
        print(" EAMF-Net-Single 模型配置")
        print(" 消融实验: 有单层分类器，交叉注意力机制，MLP特征融合，没有多层分类器")
        print("=" * 60)
        print(f" 模型参数:")
        print(f"   • hidden_dim:     {self.model_config['hidden_dim']}")
        print(f"   • mlp_hidden_dim: {self.model_config['mlp_hidden_dim']}")
        print(f"   • num_heads:      {self.model_config['num_heads']}")
        print(f"   • dropout_rate:   {self.model_config['dropout_rate']}")
        print(f"   • num_labels:     {self.model_config['num_labels']}")
        print(f" 模型组件:")
        print(f"   • BERT模型: {self.model_config['model_names'][0]}")
        print(f"   • BERT模型: {self.model_config['model_names'][1]}")
        print(f"   • BERT模型: {self.model_config['model_names'][2]}")
        print(f"   • 分类器：单层分类器")
        print(f"   • 注意力机制: 自注意力 + 三重交叉注意力")
        print(f"   • 融合方式: MLP深度融合 + 残差连接")
        print(f"   • 使用自注意力与交叉注意力的特征平均融合")
        print("=" * 60 + "\n")


    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False):
        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 交叉注意力
        cross_attended_1, cross_weights_1 = self.cross_attention_1(
            feat1.unsqueeze(1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_2, cross_weights_2 = self.cross_attention_2(
            feat2.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat3.unsqueeze(1)], dim=1)
        )
        cross_attended_3, cross_weights_3 = self.cross_attention_3(
            feat3.unsqueeze(1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1),
            torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        )

        # 3. 融合自/交叉注意力特征
        attended_feat1 = (feat1 + cross_attended_1.squeeze(1)) / 2
        attended_feat2 = (feat2 + cross_attended_2.squeeze(1)) / 2
        attended_feat3 = (feat3 + cross_attended_3.squeeze(1)) / 2

        # 4. 特征拼接
        concat_features = torch.cat([attended_feat1, attended_feat2, attended_feat3], dim=1)

        # 5. MLP特征融合
        mlp_fused = self.mlp_fusion(concat_features)
        mlp_fused = self.layer_norm2(mlp_fused + concat_features)
        mlp_fused = self.dropout(F.gelu(mlp_fused))

        # 6. 单层分类
        logits = self.classifier(mlp_fused)

        # 返回模型权重
        if return_weights:
            # 提取交叉注意力权重
            cross_weight_1 = cross_weights_1.mean(dim=1).squeeze(1)
            cross_weight_2 = cross_weights_2.mean(dim=1).squeeze(1)
            cross_weight_3 = cross_weights_3.mean(dim=1).squeeze(1)
            cross_weights = torch.cat([cross_weight_1, cross_weight_2, cross_weight_3], dim=1)  # [batch, 3]

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            if return_weights:
                return loss, logits, cross_weights
            return loss, logits
        else:
            if return_weights:
                return None, logits, cross_weights
            return None, logits
