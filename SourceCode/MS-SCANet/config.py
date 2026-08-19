import torch
import os

LOAD_MODEL = False
SAVE_MODEL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
IN_CHANS=3
EMBED_DIMS=256
NUM_HEADS=16
MLP_RATIOS=4
QKV_BIAS=True
DROP_RATE=0.4
ATTN_DROP_RATE=0.1
DROP_PATH_RATE=0.1
LEARNING_RATE_P = 1e-4
LEARNING_RATE_F = 1e-5
NUM_EPOCHS = 600
CKPT_F = 'iqa_final_{}.pth'
CKPT_P = 'iqa_pretrain_koniq10k_{}.pt'

DATASET = 'koniq'
MODEL_PATH = 'save_models'
MODEL_PATH_P = 'save_models_pretrain'
MODEL_PATH_F = 'save_models_score'
PATCH_SIZES = [16, 32]
PATCH_SIZE=16  
TRAIN_PATCH_NUM = 1  
TEST_PATCH_NUM = 1
SEED = 0
EXP_CNT = 10

IMG_SIZE = (224, 224)  # Make sure this matches the input size your model expects

RESULTS_PATH = os.getcwd()
DATA_PATH = {
    'koniq': r'./KonIQ'  # Update paths as necessary
}
