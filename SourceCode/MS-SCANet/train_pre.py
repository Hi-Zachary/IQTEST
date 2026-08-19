import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os
import csv

from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import DataLoader
from ms_scanet import MultiBranchAttentionTransformer
from utils import *
import config


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

    print('Training and Testing on <{}> dataset'.format(config.DATASET.upper()))

    if not os.path.exists(config.MODEL_PATH_P):
        os.mkdir(config.MODEL_PATH_P)

    if config.SEED != 0:
        print('SEED = {}'.format(config.SEED))
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    total_num_images = img_num[config.DATASET]
    random.shuffle(total_num_images)

    # Split data for training, validation, and testing
    num_train = int(0.8 * len(total_num_images))
    num_val = int(0.1 * len(total_num_images))

    train_index = total_num_images[:num_train]
    val_index = total_num_images[num_train:num_train + num_val]
    test_index = total_num_images[num_train + num_val:]

    dataloader_train = DataLoader(config.DATASET,
                                  folder_path[config.DATASET],
                                  train_index,
                                  config.PATCH_SIZES,
                                  config.TRAIN_PATCH_NUM,
                                  config.BATCH_SIZE,
                                  istrain=True).get_data()
    
    dataloader_val = DataLoader(config.DATASET, 
                                 folder_path[config.DATASET], 
                                 val_index, 
                                 config.PATCH_SIZES, 
                                 config.TRAIN_PATCH_NUM, 
                                 config.BATCH_SIZE, 
                                 istrain=False).get_data()

    device = torch.device(config.DEVICE)

    model = MultiBranchAttentionTransformer(img_size=config.IMG_SIZE,
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

    loss_fn = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_P, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    if config.LOAD_MODEL:
        load_path = os.path.join(config.MODEL_PATH_P, config.CKPT_P.format('latest'))
        load_checkpoint(load_path, model, optimizer, lr=config.LEARNING_RATE_P)

    model.train()
    best_plcc = 0
    best_srocc = 0
    no_improvement_epochs = 0

    csv_file = os.path.join(config.RESULTS_PATH, 'epoch_losses_cv.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

    total_epochs = config.NUM_EPOCHS
    for epoch in range(total_epochs):
        train_losses = []
        print(f'+====================+ Training Epoch: {epoch} +====================+')
        loop = tqdm(dataloader_train)

        for batch_idx, (dist, rating) in enumerate(loop):
            dist = dist.to(device).float()
            rating = rating.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            prediction = model(dist)

            # Ensure prediction shape is [B, 1]
            assert prediction.shape == rating.shape, f"Prediction shape {prediction.shape} does not match rating shape {rating.shape}"

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

            train_losses.append(total_loss.item())
            loop.set_postfix(loss=total_loss.item())

        train_plcc, train_srocc = calc_coefficient(dataloader_train, model, device)
        print(f'Epoch {epoch} Training Loss: {np.mean(train_losses)}, Training PLCC: {train_plcc:.4f}, Training SROCC: {train_srocc:.4f}')

        model.eval()
        val_losses = []
        print(f'+====================+ Validation Epoch: {epoch} +====================+')
        val_loop = tqdm(dataloader_val)

        with torch.no_grad():
            for dist, rating in val_loop:
                dist = dist.to(device).float()
                rating = rating.to(device).float().unsqueeze(1)

                prediction = model(dist)
                loss = loss_fn(prediction, rating)

                val_losses.append(loss.item())
                val_loop.set_postfix(loss=loss.item())

        val_plcc, val_srocc = calc_coefficient(dataloader_val, model, device)
        print(f'Epoch {epoch} Validation Loss: {np.mean(val_losses)}, Validation PLCC: {val_plcc:.4f}, Validation SROCC: {val_srocc:.4f}')

        # Save the epoch losses incrementally
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, np.mean(train_losses), np.mean(val_losses)])

        if config.SAVE_MODEL and epoch >= 200 and epoch % 10 == 0:
            save_path = os.path.join(config.MODEL_PATH_P, config.CKPT_P.format(epoch))
            save_checkpoint(model, optimizer, filename=save_path)

        if epoch % 20 == 0:
            embeddings, labels = extract_embeddings(dataloader_val, model, device)
            visualize_embeddings(embeddings, labels, method='PCA', epoch=epoch, save_dir=config.RESULTS_PATH)
            visualize_embeddings(embeddings, labels, method='t-SNE', epoch=epoch, save_dir=config.RESULTS_PATH)

if __name__ == '__main__':
    train_fn()
