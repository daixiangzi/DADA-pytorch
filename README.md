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

|Method | 400 | 600 | 800 | 1000 |
|-------|:-----:|:-----:|:-----:|:-----:|
 |DADA | 61.7 | 66.4 | 69.9 | 72.4 |
 |DADA_augmented | - | - | - | - |
 ## Notice
 i remove weight_norm,because it cause bad performance,when i add weight_norm.  
