# config.py

class CFG:
    debug = False
    print_freq = 100
    num_workers = 4
    model_name = 'resnext50_32x4d'
    size = 256
    # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    scheduler = 'CosineAnnealingWarmRestarts'
    epochs = 10
    factor = 0.2  # ReduceLROnPlateau
    patience = 4  # ReduceLROnPlateau
    eps = 1e-6  # ReduceLROnPlateau
    T_max = 10  # CosineAnnealingLR
    T_0 = 10  # CosineAnnealingWarmRestarts
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 32
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 5
    target_col = 'label'
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]


TRAIN_PATH = '../data/cassava-leaf-disease-classification/train_images'
TEST_PATH = '../data/cassava-leaf-disease-classification/test_images'
