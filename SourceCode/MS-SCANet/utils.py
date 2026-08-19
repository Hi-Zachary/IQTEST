from sklearn.decomposition import PCA
import torch
import torch.optim as optim
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import config
import ms_scanet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
# ========================================================
# load pretrain model
# import os
# import model_pretrain
# load_path = config.CKPT_P.format('30')
# load_path = os.path.join(config.MODEL_PATH_P, load_path)
# ckpt = torch.load(load_path, map_location=config.DEVICE)
# model_err = model_pretrain.vit_IQAModel().to(config.DEVICE)
# model_err.load_state_dict(ckpt['state_dict'])
# model_err.requires_grad_(False)
# ========================================================
def calc_coefficient(dataloader, model, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for dist, rating in dataloader:
            dist = dist.to(device).float()
            rating = rating.to(device).float().unsqueeze(1)

            prediction = model(dist)
            predictions.append(prediction.cpu().numpy())
            targets.append(rating.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    plcc = np.corrcoef(predictions[:, 0], targets[:, 0])[0, 1]
    srocc = spearmanr(predictions[:, 0], targets[:, 0])[0]
    
    return plcc, srocc


def cross_branch_consistency_loss(branch1_features, branch2_features, alpha=0.5):
    branch1_mean = branch1_features.mean(dim=1)
    branch2_mean = branch2_features.mean(dim=1)
    loss = F.mse_loss(branch1_mean, branch2_mean)
    return alpha * loss

import torch.nn.functional as F

def adaptive_pooling_consistency_loss(original_features, pooled_features, alpha=0.5):
    """
    Adaptive Pooling Consistency Loss.
    original_features: original features before pooling, shape (batch_size, num_patches, embed_dim)
    pooled_features: features after adaptive pooling, shape (batch_size, embed_dim, H, W)
    alpha: weight for the consistency term
    """
    B, N, E = original_features.shape
    _, E_pooled, H, W = pooled_features.shape

    # Ensure that pooled_features has correct dimensions
    assert E == E_pooled, "Embedding dimension mismatch between original and pooled features"

    # Calculate the target size for interpolation
    target_size = int(N ** 0.5)  # Assuming N is a perfect square

    # Upsample the pooled features to match the original features' size
    pooled_features_upsampled = F.interpolate(pooled_features, size=(target_size, target_size))
    pooled_features_reshaped = pooled_features_upsampled.view(B, E, -1).transpose(1, 2)  # [B, N, E]

    loss = F.mse_loss(original_features, pooled_features_reshaped)
    return alpha * loss


def save_checkpoint(model, optimizer, filename="model_checkpoint.pt"):
    """
    Save model and optimizer states.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for dist, rating in dataloader:
            dist = dist.to(device).float()
            rating = rating.to(device).float().unsqueeze(1)

            emb = model(dist, return_embeddings=True)
            embeddings.append(emb.cpu().numpy())
            labels.append(rating.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels


def visualize_embeddings(embeddings, labels, method='PCA', epoch=None, save_dir='results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if method == 'PCA':
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(embeddings)
        title = 'PCA of Embeddings'
        filename = os.path.join(save_dir, f'pca_epoch_{epoch}.png')
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        transformed = tsne.fit_transform(embeddings)
        title = 't-SNE of Embeddings'
        filename = os.path.join(save_dir, f'tsne_epoch_{epoch}.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load model and optimizer states from a checkpoint file.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == '__main__':
    # Device configuration (Assuming a CUDA device for this example, fallback to CPU if not available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize mock model and move it to the configured device
    model = dual_branch_attention_em().to(device)
    # Initialize a optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    checkpoint_file = 'model_checkpoint.pt'

    # Load the model checkpoint
    load_checkpoint(checkpoint_file, model, optimizer, lr=0.001)
    print("Checkpoint loaded.")

    # Demonstrate learning rate adjustment
    current_epoch = 10  # Assume current epoch
    optimizer = lr_scheduler(optimizer, current_epoch)
    print("Learning rate adjusted.")

    # demonstrate saving the current state as a new checkpoint
    save_checkpoint(model, optimizer, filename="updated_model_checkpoint.pt")
    print("Updated checkpoint saved.")


