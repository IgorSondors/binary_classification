import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from interface import get_query_emb_batch, load_model, cosine_similarity_batch
from typing import Tuple, List, Dict, Union, Any
from colbert.modeling.checkpoint import Checkpoint

class Encoder(nn.Module):
    def __init__(self, input_dim=(32, 768), latent_dim=128):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim[0] * input_dim[1], 1024)  # Flatten input
        self.fc2 = nn.Linear(1024, 256)
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the matrix to a vector
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=(32, 768)):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, output_dim[0] * output_dim[1])
        self.output_dim = output_dim

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h)).view(-1, self.output_dim[0], self.output_dim[1])  # Reshape to original

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

class SiameseNetwork(nn.Module):
    def __init__(self, latent_dim=128):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, z1, z2):
        h1 = F.relu(self.fc1(z1))
        h2 = F.relu(self.fc1(z2))
        diff = torch.abs(h1 - h2)
        out = F.relu(self.fc2(diff))
        return torch.sigmoid(self.fc3(out)).squeeze(1)  # Ensure this line produces a shape of [batch_size]


def get_query_emb(sentences: List[str], checkpoint: Checkpoint, batch_size: int) -> np.ndarray:
    """
    Generate embeddings for a list of sentences using the provided checkpoint.

    Args:
        sentences (List[str]): A list of sentences for which embeddings need to be generated.
        checkpoint (Checkpoint): The checkpoint object used for generating embeddings.
        batch_size (int): The batch size to use during inference.

    Returns:
        np.ndarray: An array of embeddings for the input sentences.
    """
    return checkpoint.queryFromText(sentences, bsize=batch_size).to("cpu").numpy()

def get_query_emb_batch(sentences: List[str], checkpoint: Checkpoint, batch_size: int, batch_size2: int) -> np.ndarray:
    """
    Generate embeddings for a list of sentences in batches using the provided checkpoint.

    Args:
        sentences (List[str]): A list of sentences for which embeddings need to be generated.
        checkpoint (Checkpoint): The checkpoint object used for generating embeddings.
        batch_size (int): The batch size to use during inference.
        batch_size2 (int): The size of the sub-batches to split the input sentences into.

    Returns:
        np.ndarray: An array of embeddings for the input sentences. 
        Shape of the array is (len(sentences), 32, 768) for bert-base-multilingual-cased or (len(sentences), 32, 128) for colbertv2.0
    """
    embeddings_list = []
    
    for i in range(0, len(sentences), batch_size2):
        # print(f"batch: {min(i+batch_size2, len(sentences))}/{len(sentences)}")

        batch_sentences = sentences[i:i+batch_size2]
        embeddings = get_query_emb(batch_sentences, checkpoint, batch_size)
        embeddings_list.append(embeddings)

        torch.cuda.empty_cache()
    
    combined_embeddings = np.concatenate(embeddings_list, axis=0)
    return combined_embeddings

class MyDataset(Dataset):
    def __init__(self, offer_batch, true_match_batch, false_match_batch, checkpoint):
        self.offer_batch = offer_batch
        self.true_match_batch = true_match_batch
        self.false_match_batch = false_match_batch
        self.checkpoint = checkpoint

    def __len__(self):
        return len(self.offer_batch)

    def __getitem__(self, idx):
        offer_embs = get_query_emb_batch([self.offer_batch[idx]], self.checkpoint, batch_size=100, batch_size2=1000)
        true_match_embs = get_query_emb_batch([self.true_match_batch[idx]], self.checkpoint, batch_size=100, batch_size2=1000)
        false_match_embs = get_query_emb_batch([self.false_match_batch[idx]], self.checkpoint, batch_size=100, batch_size2=1000)

        y_true = np.ones(len(true_match_embs))  # Метка 1 для пар (offer_embs[i], true_match_embs[i])
        y_false = np.zeros(len(false_match_embs))  # Метка 0 для пар (offer_embs[i], false_match_embs[i])

        X = np.concatenate([offer_embs, offer_embs])#.reshape(-1,32,768)
        X_pair = np.concatenate([true_match_embs, false_match_embs])#.reshape(-1,32,768)
        y = np.concatenate([y_true, y_false])#.reshape(-1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_pair_tensor = torch.tensor(X_pair, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, X_pair_tensor, y_tensor

def vae_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def validate(vae, siamese, dataloader):
    vae.eval()
    siamese.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    mean_vae_time = 0
    mean_seamese_time = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for f, g, similarity in dataloader:
            # Process through VAE
            f, g, similarity = f.reshape(-1,32,768), g.reshape(-1,32,768), similarity.reshape(-1)

            recon_f, mu_f, log_var_f = vae(f.view(f.size(0), -1))
            recon_g, mu_g, log_var_g = vae(g.view(g.size(0), -1))

            vae_time_start = time.time()
            loss_vae_f = vae_loss(recon_f, f, mu_f, log_var_f)
            loss_vae_g = vae_loss(recon_g, g, mu_g, log_var_g)
            mean_vae_time += time.time() - vae_time_start

            siamese_time_start = time.time()
            similarity_score = siamese(vae.reparameterize(mu_f, log_var_f), vae.reparameterize(mu_g, log_var_g))
            mean_seamese_time += time.time() - siamese_time_start

            loss_siamese = F.binary_cross_entropy(similarity_score, similarity.view(-1))

            loss = loss_vae_f + loss_vae_g + loss_siamese

            # Update total validation loss
            total_loss += loss.item()

            # Accuracy calculation
            predicted_labels = (similarity_score > 0.5).float()
            correct_predictions += (predicted_labels == similarity.view(-1)).sum().item()
            total_samples += similarity.size(0)

            # Collect all predictions and actual labels for ROC AUC calculation
            all_predictions.extend(similarity_score.detach().cpu().numpy())
            all_targets.extend(similarity.cpu().numpy())


    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    roc_auc = roc_auc_score(all_targets, all_predictions)  # Compute ROC AUC
    mean_vae_time = mean_vae_time / total_samples
    mean_seamese_time = mean_seamese_time / total_samples

    return average_loss, accuracy, roc_auc, mean_vae_time, mean_seamese_time

def train(vae, siamese, train_dataloader, val_dataloader, optimizer, epochs=15):
    for epoch in range(epochs):
        vae.train()
        siamese.train()
        total_train_loss = 0
        for f, g, similarity in train_dataloader:
            f, g, similarity = f.reshape(-1,32,768), g.reshape(-1,32,768), similarity.reshape(-1)
            optimizer.zero_grad()

            recon_f, mu_f, log_var_f = vae(f)
            recon_g, mu_g, log_var_g = vae(g)

            loss_vae_f = vae_loss(recon_f, f, mu_f, log_var_f)
            loss_vae_g = vae_loss(recon_g, g, mu_g, log_var_g)

            similarity_score = siamese(vae.reparameterize(mu_f, log_var_f), vae.reparameterize(mu_g, log_var_g))
            loss_siamese = F.binary_cross_entropy(similarity_score, similarity.view(-1))

            total_loss = loss_vae_f + loss_vae_g + loss_siamese
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        average_val_loss, accuracy, roc_auc, mean_vae_time, mean_seamese_time = validate(vae, siamese, val_dataloader)

        print(f'Epoch {epoch+1}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}, Val ROCAUC: {roc_auc}, Time VAE|Seamese: {mean_vae_time}|{mean_seamese_time}')
    return vae, siamese


def load_data(pth_models, pth_offers, checkpoint):

    id_category = {
        2801: 'мобильные телефоны'
        }
    
    df_models = pd.read_csv(pth_models, sep=';')
    df_offers = pd.read_csv(pth_offers, sep=';')


    df_models = df_models[df_models['category_id'].isin(id_category.keys())].reset_index(drop=True)
    df_offers = df_offers[df_offers['category_id'].isin(id_category.keys())].reset_index(drop=True)
    df_offers = df_offers#[:10000]

    df_offers_shuffled = df_offers.sample(frac=1, random_state=42)

    # Определяем размер тестовой выборки (например, 20%)
    test_size = int(0.15 * len(df_offers_shuffled))

    # Разделяем данные на тренировочную и тестовую выборки
    df_train = df_offers_shuffled.iloc[:-test_size]
    df_test = df_offers_shuffled.iloc[-test_size:]

    # Пример: вывод размеров тренировочной и тестовой выборок
    print("Размер тренировочной выборки:", len(df_train))
    print("Размер тестовой выборки:", len(df_test))

    df = df_train.copy()
    offer_batch = list(df['name'])
    true_match_batch = list(df['true_match'])
    false_match_batch = list(df['false_match'])
    train_dataset = MyDataset(offer_batch, true_match_batch, false_match_batch, checkpoint)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    df = df_test.copy()
    offer_batch = list(df['name'])
    true_match_batch = list(df['true_match'])
    false_match_batch = list(df['false_match'])
    test_dataset = MyDataset(offer_batch, true_match_batch, false_match_batch, checkpoint)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    return train_dataloader, test_dataloader

if __name__ == '__main__':

    ckpt_pth = "/home/sondors/Documents/ColBERT_weights/2801_lr04_bsize_210_apple/none/2024-04/18/09.16.10/checkpoints/colbert-187-finish"
    experiment = "colbert-187-finish"

    doc_maxlen = 300
    nbits = 2   # bits определяет количество битов у каждого измерения в семантическом пространстве во время индексации
    nranks = 1  # nranks определяет количество GPU для использования, если они доступны
    kmeans_niters = 4 # kmeans_niters указывает количество итераций k-means кластеризации; 4 — хороший и быстрый вариант по умолчанию. 

    device = "cuda"
    checkpoint = load_model(ckpt_pth, doc_maxlen, nbits, kmeans_niters, device)

    pth_models = "/home/sondors/Documents/price/BERT_data/data/17-04-2024_Timofey/2801_offers_models_Apple.csv"
    pth_offers = '/home/sondors/Documents/price/BERT_data/data/17-04-2024_Timofey/2801_Apple_triplets_offer_model_train.csv'

    train_dataloader, test_dataloader = load_data(pth_models, pth_offers, checkpoint)

    model_vae = VAE()
    model_siamese = SiameseNetwork()

    params = list(model_vae.parameters()) + list(model_siamese.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)

    # Train the model
    vae, siamese = train(model_vae, model_siamese, train_dataloader, test_dataloader, optimizer, epochs=5)