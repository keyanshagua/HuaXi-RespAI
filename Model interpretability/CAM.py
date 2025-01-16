import torch
import torch.nn.functional as F
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, ScaleIntensity, Compose
from monai.networks.nets import resnet18
import matplotlib.pyplot as plt
import numpy as np

model_weights_path = r"your_pretrained_model.pth"
image_path = r"your_image_path.nii.gz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()
original_image = LoadImage(image_only=True)(image_path)
original_z_dim = original_image.shape[-1]
preprocess = Compose([
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((128, 128, 64))
])
image = preprocess(original_image)
image = image.unsqueeze(0).to(device)
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv3d):  # 选择卷积层
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)
        break

output = model(image)
prediction = torch.argmax(output, dim=1).item()
model.zero_grad()
output[:, prediction].backward()

if activations is None or gradients is None:
    raise ValueError("Failed to capture activations or gradients. Check hook registration.")
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])
for i in range(activations.shape[1]):
    activations[:, i, :, :, :] *= pooled_gradients[i]
heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()
heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(128, 128, 64), mode='trilinear', align_corners=False).squeeze().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
slice_index = heatmap.shape[-1] // 2  
resized_z_dim = 64
scale_factor = original_z_dim / resized_z_dim
original_slice_index = int(slice_index * scale_factor)
original_image_slice = original_image[:, :, original_slice_index]

def apply_lung_window(image_slice, window_center=-400, window_width=1500):
    min_value = window_center - (window_width / 2)
    max_value = window_center + (window_width / 2)
    windowed_image = np.clip(image_slice, min_value, max_value)
    windowed_image = (windowed_image - min_value) / (max_value - min_value)
    return windowed_image

lung_window_image_slice = apply_lung_window(original_image_slice)
heatmap_slice = heatmap[:, :, slice_index]
lung_window_resized = F.interpolate(torch.tensor(lung_window_image_slice).unsqueeze(0).unsqueeze(0), size=(395, 288), mode='bilinear', align_corners=False).squeeze().numpy()
heatmap_resized = F.interpolate(torch.tensor(heatmap_slice).unsqueeze(0).unsqueeze(0), size=(395, 288), mode='bilinear', align_corners=False).squeeze().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(lung_window_resized, cmap='gray', extent=(0, 288, 0, 395))
plt.imshow(heatmap_resized, cmap='jet', alpha=0.3, extent=(0, 288, 0, 395))
plt.axis('off')
plt.show()
