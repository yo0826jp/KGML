# RDFDNN
# INTRODUCTION
Predicting relations in Knowledge Graph by Multi-Label Deep Neural Network.


# EXECUTION
## Training
```
python train_kgml.py [option] [dataset]
```
### Dataset
FB15k, WN18 or WD40k.
### Option
-d1 | Hyperparameter dim1. Integer. Default value is 100.
-d2 | Hyperparameter dim2. Integer. Default value is 100.
-a | Hyperparameter alpha. Float in [0,1]. Default value is 1.0.
### Example
```
python train_kgml.py WD40k
python train_kgml.py -d1 50 -d2 50 -a 0.5 FB15k
```

## Test
```
python test_kgml.py [model] [dataset]
```
### Example
```
python test_kgml.py kgml_WD40k.h5 WD40k
```

# DATA
FB15k and WN18 are the same as the ones used in the paper "Translating Embeddings for Modeling Multi-relational Data (2013).".  
https://everest.hds.utc.fr/doku.php?id=en:transe
They are available at data.zip in https://github.com/thunlp/KB2E.

# Experiment Environment
- tensorflow 1.4.1 
- Keras 2.1.2 
- CUDA 8.0.61
- Cudnn 6.0.20
