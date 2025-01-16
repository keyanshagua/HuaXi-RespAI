# West-China-Hospital-RespAI
West China Respiratory Chronic Disease Series Research Code library / 华西呼吸慢病系列研究代码库
![image](https://github.com/user-attachments/assets/53706556-56f3-4bd3-bc34-dd3374af4d85)
# Preface
- **Motivation**  
Chronic Obstructive Pulmonary Disease (COPD) is a leading cause of morbidity and mortality worldwide. Early and accurate detection is crucial for improving patient outcomes.

- **Problem Statement**  
We aim to explore optimal neural network architectures, fine-tune training parameters, ensure interpretability, and refine preprocessing methods—addressing COPD-specific challenges while balancing classification performance with clinical transparency.

- **Approach**  
This project compares multiple architectures (CNN variants, vision transformer networks, etc.) to determine the best-performing model for COPD classification. Interpretability techniques (e.g., Grad-CAM, Occlusion Sensitivity) are also employed to clarify how each model reaches its decisions.

# Model Architectures
![image](https://github.com/user-attachments/assets/0dfaec3f-93f8-493a-86ad-7ab451dd8b5d)
    (a) ResNet configuration (b) DenseNet configuration (c) ViT configuration

# Interpretability
## CAM
![image](https://github.com/user-attachments/assets/f24e1c0d-e63e-4906-a778-b879aa0cb457)

## Grad-CAM and Occlusion Sensitivity
![image](https://github.com/user-attachments/assets/2648c1f0-f84a-4a5b-9e5d-529183be41df)

# Installation & Usage
**1. Setup**

`git clone https://github.com/keyanshagua/West-China-Hospital-RespAI.git`

 `pip install -r requirements.txt`
 
**2. Get the MONAI Docker Image**

`docker pull projectmonai/monai:latest`
