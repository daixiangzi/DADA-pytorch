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
Keras
# Run
python3 train.py  
Default Set:config.py  
# Results  
Best Acc  

|Method | 400 | 600 | 800 | 1000 |
|-------|:-----:|:-----:|:-----:|:-----:|
 |DADA | 63.0 | 67.6 | 71.2 | 73.3 |
 |DADA_augmented | - | - | - | - |
# Train process of Gen img in 200 epochs(DADA)  
** one row represent one class (100 fixed noise)**  
![image](./img/result.gif)  
 ## Notice
 DADA_augmented result will be released later  
 i remove weight_norm,because it cause bad performance,when i add weight_norm.  
