# Emotions Detector

## Preliminary Steps

1. Download the dataset from the following link: [emotions-detector.zip](https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip)

2. Extract the dataset and move the CSV files into the "data" folder.

# Ingredients and hyper-parameters used for ResNet-50 training by Ross Wightman, Hugo Touvron, and Hervé Jégou

Train res: 160
Test res: 224
Epochs: 100
Batch size: 2048
Optimizer: LAMB
LR: 8 X 10-3
decay rate: cosine
Weight decay: 0.02
warmup epochs: 5
H. flip: yes
RRC: yes
Rand Augment 6/0.5
Test crop ratio: 0.95
Mixed precision: yes

# What i will do

train res: 224
test res: 224
epochs: 100
batch size: 128
Oprimizer: AdamW
lr: 8 x 10-3
decay rate: cosine
Weight decay: 0.02
warmup epochs: 5
H. flip: yes
RRC: yes
Rand Augment 6/0.5
Test crop ratio: 0.95
Mixed precision: yes

# Links to read

The project
https://github.com/01-edu/public/blob/master/subjects/ai/emotions-detector/README.md

the datset
https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview

DeiT
https://arxiv.org/abs/2012.12877

Resnet
https://arxiv.org/pdf/1512.03385

He initialization whihc conv2d uses
https://arxiv.org/abs/1502.01852

ResNet strikes back: An improved training procedure in timm
https://arxiv.org/abs/2110.00476
