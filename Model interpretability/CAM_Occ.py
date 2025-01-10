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
)
from monai.data import DataLoader, PersistentDataset
from monai.networks.nets import resnet18
from monai.visualize import GradCAM, OcclusionSensitivity

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印设备信息
print(f"Using device: {device}")

# 加载JSON文件
json_path = "/work/home/ac609hjkef/liyp/Direct_classification_2/Data_split/labels_version_1.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# 创建标签映射字典
label_mapping = {'nor': 0, 'copd': 1}

# 提取训练、验证和测试数据，并将字符串标签映射为整数
train_data = data.get('train', [])
val_data = data.get('val', [])
test_data = data.get('test', [])

train_files = [{"image": item['path'], "label": label_mapping[item['label']]} for item in train_data]
val_files = [{"image": item['path'], "label": label_mapping[item['label']]} for item in val_data]
test_files = [{"image": item['path'], "label": label_mapping[item['label']]} for item in test_data]

# 定义数据转换流水线
train_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(320, 320, 64)),
    RandRotate90d(keys=["image"], prob=0.8),
    ToTensord(keys=["image"]),
])

# 定义缓存目录
root_dir = "/work/home/ac609hjkef/liyp/Direct_classification_2/Res18"
persistent_cache = os.path.join(root_dir, "persistent_cache")

# 确保缓存目录存在
os.makedirs(persistent_cache, exist_ok=True)

# 加载数据集并创建数据加载器
train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=persistent_cache)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

# 定义输入图像大小
win_size = (320, 320, 64)

# 加载训练好的ResNet18模型
model_3d = resnet18(
    spatial_dims=3,        # 3D数据
    n_input_channels=1,    # 输入通道数
    num_classes=2          # 输出类别数
).to(device)

# 打印模型架构以确认层名称
print("ResNet18模型架构:")
print(model_3d)

model_path = "best_metric_model_resnet18.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

model_3d.load_state_dict(torch.load(model_path, map_location=device))
model_3d.eval()

# 打印模型的所有模块名称
print("\n模型的所有模块名称:")
for name, module in model_3d.named_modules():
    print(name)

# 初始化GradCAM，指定目标层名称
# 根据模型的模块名称，选择目标层
target_layer = 'layer4.1.conv2'  # 修改为层的名称字符串

print(f"\nSelected target layer: {target_layer}")
cam = GradCAM(nn_module=model_3d, target_layers=[target_layer])

# 打印特征图尺寸
print(
    "Original feature map size:",
    cam.feature_map_size([1, 1] + list(win_size), device),
)
print("Upsampled feature map size:", [1, 1] + list(win_size))

# 初始化Occlusion Sensitivity，减少掩膜大小
occ_sens = OcclusionSensitivity(nn_module=model_3d, mask_size=8, n_batch=1)

# 定义感兴趣的切片
the_slice = win_size[-1] // 2  # 使用图像的深度维度
occ_sens_b_box = [-1, -1, -1, -1, the_slice - 1, the_slice]

# 设置随机状态以保证可重复性
np.random.seed(42)

# 指定要加载的图像路径（请根据实际路径修改）
TARGET_IMAGE_PATH = "/work/home/ac609hjkef/liyp/Direct_classification_2/copd/Lung_0001183293_2018-12-02_image_cropped.nii.gz"

# 查找该路径在 train_files 中的索引
SPECIFIC_INDEX = next((i for i, item in enumerate(train_files) if item['image'] == TARGET_IMAGE_PATH), None)

if SPECIFIC_INDEX is None:
    raise ValueError(f"Image path {TARGET_IMAGE_PATH} not found in training dataset.")

# 加载指定的训练数据
data_item = train_ds[SPECIFIC_INDEX]
image = data_item["image"].to(device).unsqueeze(0)  # 添加批次维度
label = data_item["label"]  # 标签是表示类别的整数
y_pred = model_3d(image)
pred_label = y_pred.argmax(1).item()

# 仅显示实际标签为COPD且预测正确的图像
if label != 1 or pred_label != 1:
    raise ValueError(f"The specified image at index {SPECIFIC_INDEX} does not have label 'COPD' or was not correctly predicted.")

# 提取感兴趣的切片图像
img = image.detach().cpu().numpy()[0, 0, ..., the_slice]

# 构建标题信息
name = f"实际: COPD"
name += f"\n预测: COPD"
name += f"\nCOPD 概率: {y_pred[0,1]:.3f}"
name += f"\nNormal 概率: {y_pred[0,0]:.3f}"

# 运行GradCAM
cam_result = cam(x=image, class_idx=1)  # 指定类别索引为1（COPD）
cam_result = cam_result[..., the_slice].cpu().detach().numpy()

# 运行Occlusion Sensitivity
occ_result, _ = occ_sens(x=image, b_box=occ_sens_b_box)
occ_result = occ_result[0, pred_label][None, None, ..., -1].cpu().detach().numpy()

# 设置更大的图形尺寸
fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="white")

# 定义要显示的图像及其标题
images = [img, cam_result[0,0], occ_result[0,0]]
titles = [name, "GradCAM", "Occlusion Sensitivity"]
cmaps = ["gray", "jet", "jet"]

for idx, (im, title, cmap) in enumerate(zip(images, titles, cmaps)):
    ax = axes[idx]
    im_show = ax.imshow(im, cmap=cmap)
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    fig.colorbar(im_show, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()

# 保存图像为PNG文件
output_path = 'gradcam_specific_image.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存为 {output_path}")

# 显示图像（如果需要）
plt.show()
