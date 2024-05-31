# Emotions Detector

## Preliminary Steps

1. Download the dataset from the following link: [emotions-detector.zip](https://assets.01-edu.org/ai-branch/project3/emotions-detector.zip)
2. Extract the dataset and move the CSV files into the "data" folder.
3. To start TensorBoard, use the following command:
   ```bash
   tensorboard --logdir=runs
   ```

## Model

The model used is a ResNeXt architecture with a Convolutional Block Attention Module (CBAM).

- [ResNeXt Paper](https://arxiv.org/pdf/1611.05431)
- [CBAM Paper](https://arxiv.org/abs/1807.06521)

![ResNeXt model with CBam](model.png)

## Model Description

### Initialization

The weights for the convolutional and linear layers are initialized using He initialization (Kaiming Normal).

- [He Initialization Paper](https://arxiv.org/abs/1502.01852)

The initial loss is approximately 1.9, which is appropriate since the initial learning rate should be log(1/C) where C is the number of classes (C = 7).

## Hyperparameters

- Train resolution: 224
- Test resolution: 224
- Epochs: 150
- Batch size: 128
- Optimizer: AdamW
- Learning rate (lr): 1e-3
- Weight decay: 0.03
- Label smoothing: 0.1

## Useful Links

- [Project Description](https://github.com/01-edu/public/blob/master/subjects/ai/emotions-detector/README.md)
- [Dataset](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview)
- [DeiT Paper](https://arxiv.org/abs/2012.12877)
- [DeiT Video](https://www.youtube.com/watch?v=viClVMxiwI0)
- [ResNet Paper](https://arxiv.org/pdf/1512.03385)
- [He Initialization Paper](https://arxiv.org/abs/1502.01852)
- [ResNet Strikes Back: Improved Training Procedure in TIMM](https://arxiv.org/abs/2110.00476)
- [CBAM Paper](https://arxiv.org/abs/1807.06521)
- [ResNeXt Paper](https://arxiv.org/pdf/1611.05431)

The model can be found at
https://drive.google.com/file/d/11ce8kRUZ8rETBFT84_TuasEm7MWgLkVv/view?usp=drive_link
