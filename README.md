## Histopathologic Cancer Detection

----
https://www.kaggle.com/c/histopathologic-cancer-detection

Task:
Identify metastatic cancer in small image patches taken from larger digital pathology scans.

Architecture:
CNN base network (resnet or densenet here) + pooling (concat avg and max pooling)  + classifier

Solution:
Finetuning the whole network with one-cycle learning rate schedule (discriminative for each layer group). Inference with TTA (time/performance trade-off). Ensemble in 2 ways: 1. average outputs; 2. average weights of networks (similar to SWA), both from different data split and/or network architectures. Pseudo-labeling also boost much of the performance.

### Dependencies
pytorch v1.1.0, fastai v1
