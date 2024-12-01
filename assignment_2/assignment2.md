## Name

Ting-Hsuan Lien, Tzu-Hsiang Kao, Cheng-Han Shen

## Train/val/test split percentage

70/20/10

## Selected model(s):

SVM, KNN, MLP

## Hyperparameter tuning or model selection: 

- Model selection:

  SVM

- Hyperparameter tuning

  KNN, MLP

## Parameters and model selection

- for SVM kernel: 
  - linear
  - rbf
  - poly
  - sigmoid
  - precomputed
- for KNN:
  - K= 1, 3, 5, 7, 9
- for MLP:
  - hidden layer amount= 1, 2, 3

## Best model/hyperparameter

- SVM: kernel = rbf
- KNN: K = 7
- MLP: 2 hidden layer

## Performance of best model on your test set (accuracy):

- SVM: 59.83%

- KNN: 58.97%

- MLP: 58.12%

## Contributions:

Ting-Hsuan Lien: preprocessing, implement SVM, postprocessing

Tzu-Hsiang Kao: implement MLP, write PDF report

Cheng-Han Shen: implement KNN