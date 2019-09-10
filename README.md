## Histopathologic Cancer Detection

----
https://www.kaggle.com/c/histopathologic-cancer-detection

Task:
Identify metastatic cancer in small image patches taken from larger digital pathology scans.

Architecture:
CNN base network (resnet or densenet here) + pooling (concat avg and max pooling)  + classifier

Solution:
Fine-tuning the whole network with one-cycle learning rate schedule. Inference with TTA and ensemble outputs from different network architectures.

### Dependencies
Pytorch 1.0.1
