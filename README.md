# Cassava Leaf Disease Classification

## Model

* [x] ResNeXt50_32x4d

* [ ] Vision Transformer (ViT) (to be continued ...)

## How to run

### Environment
```
conda env create --name cassava python=3.9
conda activate cassava
pip install -r requirements.txt
```

### Reproducing solution

1. Downloading [competition data](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data) and adding it into the `data/` folder.

2. Use `python src/main.py --mode train` to run training.

3. Use `python src/main.py --mode inference` to run inferencing.
