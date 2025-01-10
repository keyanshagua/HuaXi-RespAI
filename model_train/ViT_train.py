import os
import json
import copy
from typing import List, Dict
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
)
from monai.data import Dataset
from monai.utils import set_determinism

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 导入您自定义的ViT模型
from vit_model import ViT  # 确保vit_model.py在同一目录下或在PYTHONPATH中

# 设置随机种子以确保结果可复现
set_determinism(seed=42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数
num_epochs = 200
batch_size = 2  # 根据硬件资源调整
learning_rate = 5e-5  # 调整后的学习率
num_classes = 2
patience = 20  # 早停的耐心值

# 加载 JSON 文件
json_path = "/work/home/ac609hjkef/liyp/Direct_classification_2/Data_split/labels_version_1.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# 创建标签映射字典
label_mapping = {'nor': 0, 'copd': 1}

# 提取训练、验证和测试集的数据，并将字符串标签映射为整数
train_data = data.get('train', [])
val_data = data.get('val', [])
test_data = data.get('test', [])

train_files = [
    {"image": item['path'], "label": label_mapping[item['label']]}
    for item in train_data
]
val_files = [
    {"image": item['path'], "label": label_mapping[item['label']]}
    for item in val_data
]
test_files = [
    {"image": item['path'], "label": label_mapping[item['label']]}
    for item in test_data
]

# 检查标签分布
print("训练集标签分布:", Counter([item['label'] for item in train_files]))
print("验证集标签分布:", Counter([item['label'] for item in val_files]))
print("测试集标签分布:", Counter([item['label'] for item in test_files]))

# 定义数据转换流程（添加数据增强）
train_transforms = Compose([
    LoadImaged(keys=["image"]),  # 仅加载 "image"
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotate90d(keys=["image"], prob=0.5, max_k=3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 1.5)),
    RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
    ToTensord(keys=["image", "label"]),  # 将 "image" 和 "label" 转换为张量
])

val_transforms = Compose([
    LoadImaged(keys=["image"]),  # 仅加载 "image"
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    ToTensord(keys=["image", "label"]),
])

test_transforms = Compose([
    LoadImaged(keys=["image"]),  # 仅加载 "image"
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    ToTensord(keys=["image", "label"]),
])

# 定义缓存目录（可选）
root_dir = "/work/home/ac609hjkef/liyp/Direct_classification_2/ViT_train"
persistent_cache = os.path.join(root_dir, "persistent_cache")

# 创建Dataset
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)
test_ds = Dataset(data=test_files, transform=test_transforms)

# 创建DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 实例化ViT模型
model = ViT(
    in_channels=1,                # 单通道
    img_size=(320, 320, 64),      # 根据Resized后的尺寸
    patch_size=(16, 16, 16),      # 块大小，可根据需求调整
    hidden_size=768,
    mlp_dim=3072,
    num_layers=12,
    num_heads=12,
    proj_type='conv',
    pos_embed_type='learnable',
    classification=True,
    num_classes=num_classes,
    dropout_rate=0.1,
    spatial_dims=3,
    post_activation=None,  # 移除激活函数
    qkv_bias=True,
    save_attn=False
)

model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# 定义学习率调度器（使用ReduceLROnPlateau）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# 定义评估指标
def calculate_metrics(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1).cpu().numpy()
    truths = y_true.cpu().numpy()
    precision = precision_score(truths, preds, average='binary', zero_division=0)
    recall = recall_score(truths, preds, average='binary', zero_division=0)
    f1 = f1_score(truths, preds, average='binary', zero_division=0)
    return precision, recall, f1

# 训练和验证循环
best_metric = -1
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

# 创建保存模型的目录
save_dir = os.path.join(root_dir, "models")
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 10)
    
    # 每个epoch有训练和验证阶段
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # 设置为训练模式
            dataloader = train_loader
        else:
            model.eval()   # 设置为评估模式
            dataloader = val_loader
        
        running_loss = 0.0
        running_corrects = 0
        running_precision = 0.0
        running_recall = 0.0
        running_f1 = 0.0
        num_samples = 0
        
        # 迭代数据
        for batch in tqdm(dataloader, desc=f"{phase}"):
            inputs = batch["image"].to(device, dtype=torch.float)
            labels = batch["label"].to(device, dtype=torch.long)
            
            # 零梯度
            optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(phase == "train"):
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                # 计算指标
                precision, recall, f1 = calculate_metrics(outputs, labels)
                
                # 反向传播和优化
                if phase == "train":
                    loss.backward()
                    optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.argmax(outputs, dim=1) == labels.data)
            running_precision += precision * inputs.size(0)
            running_recall += recall * inputs.size(0)
            running_f1 += f1 * inputs.size(0)
            num_samples += inputs.size(0)
        
        epoch_loss = running_loss / num_samples
        epoch_acc = running_corrects.double() / num_samples
        epoch_precision = running_precision / num_samples
        epoch_recall = running_recall / num_samples
        epoch_f1 = running_f1 / num_samples
        
        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}")
        
        # 深度复制模型
        if phase == "val":
            scheduler.step(epoch_loss)  # 调整学习率
            if epoch_acc > best_metric:
                best_metric = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                # 保存当前最好的模型
                best_model_path = os.path.join(save_dir, f"best_vit_model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated and saved at epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    break
    
    # 如果早停被触发，则跳出epoch循环
    if epochs_no_improve >= patience:
        break

print(f"Best val Acc: {best_metric:.4f}")

# 加载最佳模型权重
model.load_state_dict(best_model_wts)

# 测试集评估
model.eval()
test_corrects = 0
test_precision = 0.0
test_recall = 0.0
test_f1 = 0.0
test_samples = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        inputs = batch["image"].to(device, dtype=torch.float)
        labels = batch["label"].to(device, dtype=torch.long)
        
        outputs, _ = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        
        test_corrects += torch.sum(preds == labels.data)
        test_precision_batch, test_recall_batch, test_f1_batch = calculate_metrics(outputs, labels)
        test_precision += test_precision_batch * inputs.size(0)
        test_recall += test_recall_batch * inputs.size(0)
        test_f1 += test_f1_batch * inputs.size(0)
        test_samples += inputs.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_corrects.double() / test_samples
test_precision = test_precision / test_samples
test_recall = test_recall / test_samples
test_f1 = test_f1 / test_samples

print(f"Test Acc: {test_acc:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f} F1: {test_f1:.4f}")

# 生成并打印混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

# 保存最佳模型
final_model_path = os.path.join(save_dir, "best_vit_model_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Best model saved at {final_model_path}")
