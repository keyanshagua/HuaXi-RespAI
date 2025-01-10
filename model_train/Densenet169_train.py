# 导入库
import os
import json
import monai
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandRotate90d,
    ToTensord,
    AsDiscreted,
)
from monai.data import DataLoader, PersistentDataset
from monai.config import print_config
from monai.networks.nets import DenseNet169

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印 MONAI 配置
print_config()

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

train_files = [{"image": item['path'], "label": label_mapping[item['label']]} for item in train_data]
val_files = [{"image": item['path'], "label": label_mapping[item['label']]} for item in val_data]
test_files = [{"image": item['path'], "label": label_mapping[item['label']]} for item in test_data]

# 定义数据转换流程
train_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    RandRotate90d(keys=["image"], prob=0.8),
    ToTensord(keys=["image"]),
    AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long)
])

val_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    ToTensord(keys=["image"]),
    AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long)
])

test_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    ToTensord(keys=["image"]),
    AsDiscreted(keys=["label"], to_onehot=None, dtype=torch.long)
])

# 定义缓存目录
root_dir = "/work/home/ac609hjkef/liyp/Direct_classification_2/Dense169"
persistent_cache = os.path.join(root_dir, "persistent_cache")

# 加载数据集并创建数据加载器
train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=persistent_cache)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=persistent_cache)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=True)

test_ds = PersistentDataset(data=test_files, transform=test_transforms, cache_dir=persistent_cache)
test_loader = DataLoader(test_ds, batch_size=2, num_workers=2, pin_memory=True)

# 加载 ResNet18 模型
model = DenseNet169(
    spatial_dims=3,        # For 3D data
    in_channels=1,    # Number of input channels
    out_channels=2          # Number of output classes
).to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 开始训练
val_interval = 1
max_epochs = 300
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device).view(-1)  # 修改后的代码，使用 view(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step += 1
        if step % 50 == 0:
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # 验证阶段
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device).view(-1)  # 修改后的代码
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    "best_metric_model_Densenet169.pth",
                )
            print(
                f"current epoch: {epoch + 1} current accuracy: {metric:.4f}"
                f" best accuracy: {best_metric:.4f}"
                f" at epoch {best_metric_epoch}"
            )

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# 绘制训练损失和验证准确率曲线并保存为 PNG 格式
plt.figure(figsize=(12, 5))

# 绘制训练损失
plt.subplot(1, 2, 1)
plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 绘制验证准确率
plt.subplot(1, 2, 2)
plt.plot(range(1, len(metric_values) + 1), metric_values, label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_loss_and_validation_accuracy.png")
plt.close()

# 测试阶段
model.load_state_dict(torch.load("best_metric_model_Densenet169.pth"))
model.eval()
with torch.no_grad():
    num_correct = 0.0
    metric_count = 0
    for test_data in test_loader:
        test_images = test_data["image"].to(device)
        test_labels = test_data["label"].to(device).view(-1)  # 修改后的代码
        test_outputs = model(test_images)
        value = torch.eq(test_outputs.argmax(dim=1), test_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
    test_metric = num_correct / metric_count
    print(f"Test accuracy: {test_metric:.4f}")
