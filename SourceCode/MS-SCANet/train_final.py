import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import DataLoader
from ms_scanet import MultiBranchAttentionTransformer
from utils import *
import config
import csv


def train_fn():
    folder_path = {
        'live': config.DATA_PATH['live'],
        'csiq': config.DATA_PATH['csiq'],
        'livec': config.DATA_PATH['livec'],
        'koniq': config.DATA_PATH['koniq'],
    }

    img_num = {
        'live': list(range(0, 779)),
        'csiq': list(range(0, 866)),
        'livec': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
    }

    print(f'Training and Testing on <{config.DATASET.upper()}> dataset')

    if not os.path.exists(config.MODEL_PATH):
        os.mkdir(config.MODEL_PATH)

    if config.SEED != 0:
        print(f'SEED = {config.SEED}')
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    total_num_images = img_num[config.DATASET]
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

    dataloader_train = DataLoader(config.DATASET,
                                  folder_path[config.DATASET],
                                  train_index,
                                  config.PATCH_SIZES,
                                  config.TRAIN_PATCH_NUM,
                                  config.BATCH_SIZE,
                                  istrain=True).get_data()

    dataloader_test = DataLoader(config.DATASET,
                                 folder_path[config.DATASET],
                                 test_index,
                                 config.PATCH_SIZES,
                                 config.TEST_PATCH_NUM,
                                 istrain=False).get_data()

    device = torch.device(config.DEVICE)
    model = SingleBranchAttentionTransformer(img_size=config.IMG_SIZE,
                                               patch_sizes=config.PATCH_SIZES,
                                               in_chans=config.IN_CHANS,
                                               embed_dim=config.EMBED_DIMS,
                                               num_heads=config.NUM_HEADS,
                                               mlp_ratio=config.MLP_RATIOS,
                                               qkv_bias=config.QKV_BIAS,
                                               drop_rate=config.DROP_RATE,
                                               attn_drop_rate=config.ATTN_DROP_RATE,
                                               drop_path_rate=config.DROP_PATH_RATE).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Load the pre-trained modelß
    pretrained_model_path = './save_models_pretrain/iqa_pretrain_koniq10k_500.pt'  # Update this path
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # If the model was saved with nn.DataParallel, which stores the model state with a module. prefix
            if next(iter(state_dict)).startswith('module.'):
                # Remove 'module.' prefix
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print("Pre-trained model loaded successfully.")
        else:
            print("Pre-trained model state dict not found in the checkpoint.")
    else:
        print("Pre-trained model not found. Training from scratch.")

    loss_fn = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_F, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    model.train()
    best_srocc = 0
    best_model = None
    best_optimizer = None
    epoch_losses = []
    for epoch in range(config.NUM_EPOCHS):
        losses = []
        print(f'+====================+ Training Epoch: {epoch} +====================+')
        loop = tqdm(dataloader_train)

        for batch_idx, (dist, rating) in enumerate(loop):
            dist = dist.to(device).float()
            rating = rating.to(device).float().unsqueeze(1)  # Ensure rating dimensions match

            optimizer.zero_grad()
            outputs = model(dist)

            if len(outputs) == 5:
                prediction, x1, x2, original_features, pooled_features = outputs
            else:
                prediction = outputs

            # Calculate primary loss
            loss = loss_fn(prediction, rating)

            if len(outputs) == 5:
                # Reshape original features to expected 3D format (B, N, E)
                B, E, H, W = original_features.shape
                original_features_reshaped = original_features.view(B, E, -1).transpose(1, 2)  # [B, N, E]

                # Calculate consistency losses
                cb_consistency_loss = cross_branch_consistency_loss(x1, x2, alpha=0.5)
                ap_consistency_loss = adaptive_pooling_consistency_loss(original_features_reshaped, pooled_features,
                                                                        alpha=0.5)

                total_loss = loss + cb_consistency_loss + ap_consistency_loss
            else:
                total_loss = loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(total_loss.item())
            loop.set_postfix(loss=total_loss.item())

        epoch_loss = sum(losses) / len(losses)
        epoch_losses.append(epoch_loss)
        print(f'Loss: {epoch_loss:.5f}')
        print(f'+====================+ Testing Epoch: {epoch} +====================+')

        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            for dist, rating in dataloader_test:
                dist = dist.to(device).float()
                rating = rating.to(device).float().unsqueeze(1)
                prediction = model(dist)
                all_predictions.extend(prediction.cpu().numpy())
                all_labels.extend(rating.cpu().numpy())

            # Plot scatter plot
            plt.figure(figsize=(5, 5))
            plt.scatter(all_predictions, all_labels, alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'KonIQ-10k')
            plt.grid(True)
            plt.savefig(f'scatter_epoch_{epoch}.png')
            plt.close()

            sp, pl = calc_coefficient(dataloader_test, model, device)
            print(f'SROCC: {sp:.3f}, PLCC: {pl:.3f}')

        if sp > best_srocc:
            best_srocc = sp
            best_model = model
            best_optimizer = optimizer

        print(f'BEST SROCC: {best_srocc:.3f}')
        model.train()

    coef = {'srocc': best_srocc, 'plcc': sp}
    return coef, best_model, best_optimizer, best_srocc, epoch_losses


if __name__ == '__main__':
    print("Starting training...")
    coef, best_model, best_optimizer, best_srocc, epoch_losses = train_fn()  # Capture epoch_losses here

    if config.SAVE_MODEL:
        save_path = config.CKPT_F.format(config.DATASET)
        save_path = os.path.join(config.MODEL_PATH, save_path)
        print(f"Saving best model with SROCC: {best_srocc} to {save_path}")
        save_checkpoint(best_model, best_optimizer, filename=save_path)

    # Save the results after the training is completed
    headers = list(coef.keys())
    with open(f'{config.DATASET}_Results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerow(coef)

    # Optionally, save the epoch losses for later analysis or visualization
    with open(f'{config.DATASET}_EpochLosses.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss'])
        for i, loss in enumerate(epoch_losses):
            writer.writerow([i + 1, loss])  # i+1 to denote epoch number starting from 1

    print("Training completed.")
