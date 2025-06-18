# Prioritized-Training
Reproducing code from this paper https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf


# Datasets
We need a mix of classification and regression datasets to test the prioritized training approach. 

Even though prioritized training is aimed at a scenario where data is plentiful and compute is restricted, they demonstrated on MNIST, CIFAR, and other smallish datasets. This is actually good news for us, since regression labels are a bit harder to come by than classification labels so the datasets usually end up being smaller.

## Classification Datasets and Models (to reproduce)
- ConvModel as the holdout model for all classification tasks
- [x] QMNIST 
    - 120k grayscale images at 28x28 with 10 classes
    - 2 Layer MLP with hidden size 512
    - Cross Entropy Loss
- [ ] CIFAR10 
    - 60k RGB images at 32x32 with 10 classes
    - ResNet-18 adapted for small images (no downsampling)
    - Cross Entropy Loss
- [ ] CIFAR100 
    - 60k RGB images at 32x32 with 100 classes
    - ResNet-18 adapted for small images (no downsampling)
    - Cross Entropy Loss
- [ ] CINIC10 
    - 270k 32x32 RGB images at 32x32 with 10 classes
    - ResNet-18 adapted for small images (no downsampling)
    - Cross Entropy Loss


## Regression Datasets
- [ ] COCO detection bounding boxes
    - 330k images at 640x480 with 80 classes (regression labels are bounding boxes)
    - ResNet-18
- [ ] COCO Joint keypoint 
    - 330k images at 640x480 with 34 classes (regression labels are joint keypoints)
    - ResNet-18
    - **Note**: Keypoint regression requires special augmentation handling. Standard image augmentations like horizontal flips must also transform the keypoint coordinates and swap left/right keypoint pairs. Currently, augmentation is disabled for this dataset to avoid incorrect training. 
- [ ] MPI-INF-3DHP dataset 
    - pose dataset 1.3M frames 
- [ ] AffectNet 
    - 450k annotated images at 256x256 with 8 classes (regression labels are valence and arousal)
- [ ] RAF-DB 
    - 30k annotated images at 256x256 with 7 classes (regression labels are valence and arousal)

# Metrics
There are two metrics we care about comparing across our different datasets
- accuracy of the taget model at the end of training for 5 epochs
  - we expect the target model to end up with a higher accuracy than the uniform sampling model, but we will also compare the accuracy of the holdout model at the end of training
- number of steps it took to reach some target accuracy (which will be equal to the accuracy of the uniform sampling model at the end of 5 epochs)

We will want to generate plots that look like this to compare the speedup to target accuracy:
![alt text](Mindermann22a.png)

# Models and Hyperparmeters
- "3 Layer MLP for experiments on QMNIST"
- Conv Model similar to LeNet for the holdout model
- "ResNet-18 adapted for small images for CIFAR-10/CIFAR100/CINIC-10"

They use default PyTorch hyperparameters with a 10% subsammple rate

They on to say 
> "In our default setting (Fig. 2, row 1), both the target model and IL
model have the same architecture (ResNet-18). In rows 2
and below, we instead used a small CNN similar to LeNet
as the IL model (LeCun et al., 1989). It has 21x fewer
parameters and requires 29x fewer FLOP per forward pass
than the ResNet-18. The smaller IL model accelerates training as much or more than the larger model, even though its final accuracy is far lower than the target ResNet18."

In our experiments, we will use a small CNN similar to LeNet as the IL model, and a ResNet-18 as the target model.





------

In the paper, they use a copy of the target model to calculate the loss on the whole potential training batch, but in practice many GPUs are now memory constrained and having a copy of the target model is less feasible. 

# Hyperparameters
Paper uses default PyTorch hyperparameters with a 10% subsammple rate


# Questions
- Do we need a test set for the holdout model? or can we just use the holdout training accuracy as our metric to determine when to stop training the holdout model? Ideally use 10% of holdout training data as a test set, but this is not strictly necessary.

Does the architecture matter for how well the technique works? What if we extend this to a transformer model?


# Constraints and criteria
We should be training with AdamW optimizer with the PyTorch default hyperparameters (I'm not sure if these are the same in Tensorflow keras, so we shold look this up to be sure). We should use a batch size of 64 and scale the big batch size according to the sample rate . We should also be using a batch size of 256, and a subsample rate of 10% for the prioritized training.

We will train our uniform sampling model, aka the random sampling / null hypothesis model, for 5 epochs "to convergence" as they say in the paper. The accuracy of the uniform sampling model at the end of 5 epochs will be used as the benchmark against which we can judge speedups. We train our prioritized training model for 5 epochs as well. 



We should also create a table like this one 





While our model / loss will be different between the classification and regression tasks, we want to make sure WITHIN an experiment both the target model and the holdout model are using the same loss function and we use that same loss when calculating the RHO loss. 




# Results

When using no holdout data, ie double dipping between the train and test data: we have a table showing dataset Target steps to target accuracy, Uniform samplings vs RHO-LOSS
| Dataset  | Target Acc | Uniform Epochs to Target / Accuracy | RHO-LOSS Steps / Accuracy   |
|--------- |------------|---------------------|----------  |
| CIFAR10  | 80%<br>90% | 39 <br> 177 (90.8%) | 17<br>47 (92.2%)  |
| CIFAR100 | 50%<br>65% | 47 <br> 142 (67.8%) | 22<br>87 (68.1)%  |
| CINIC10  | 70%<br>80% | 37 <br> 146 (80.1%) | 26 <br> 70 (82.1%)|

![Table](Mindermann-table2.png)

If we look at the far right column, that is what we will be testing in some different regression scenarios. 

