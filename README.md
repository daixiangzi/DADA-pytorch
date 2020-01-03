# DADA-pytorch
For cifar10 ”DADA: Deep Adversarial Data Augmentation for Extremely Low Data Regime Classification“

# Related  
Origin paper:https://arxiv.org/abs/1809.00981  
Official Implementation(Theano): https://github.com/SchafferZhang/DADA  
# Requirement  
python3.5  
pytorch 1.1.0  
cuda8.0  
torchvision  
# Run
python3 train.py  
Default Set:config.py  
# Results  
Best Acc  
|Method|400 per class|600 per class|800 per classs| 1000 per class|
|-------|-------|-------|-------|-------|
 |DADA | 61.7 | 66.4 | 69.9 | 72.4 |
 |DADA_Aug | - | - | - | - |
 
|Method |  Sub. 1 |  Sub. 2  | Sub. 3  | Sub. 4  | Sub. 5 | Sub. 6 | Sub. 7 |
|-------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
 |SVM | 71.8 | 64.5 | 69.3 | 93.0 | 77.5 | 72.5 | 68.0 |
 |CNN | 74.5 | 64.3 | 71.8 | 94.5 | 79.5 | 75.0 | 70.5 |
 |CNN-SAE | 76.0 | 65.8 | 75.3 | 95.3 | 83.0 | 79.5 | 74.5 |
 |DADA | **76.6** | **66.8** | **75.6** | **96.5** | **83.2** | **80.2** | **77.0** |
