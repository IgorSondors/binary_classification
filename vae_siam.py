import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from interface import load_model#, get_query_emb_batch
from typing import List, Dict
from colbert.modeling.checkpoint import Checkpoint

import wandb

from transformers import AdamW, get_linear_schedule_with_warmup

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
        out = self.fc2(diff)
        return self.fc3(out).squeeze(1)  # Ensure this line produces a shape of [batch_size]

def get_query_emb(sentences: List[str], checkpoint: Checkpoint, batch_size: int) -> torch.Tensor:
    with torch.no_grad():
        return checkpoint.queryFromText(sentences, bsize=batch_size)#.to("cpu").numpy()

def get_query_emb_batch(sentences: List[str], checkpoint: Checkpoint, batch_size: int, batch_size2: int) -> torch.Tensor:
    embeddings_list = []
    
    for i in range(0, len(sentences), batch_size2):
        batch_sentences = sentences[i:i+batch_size2]
        with torch.no_grad():
            embeddings = torch.tensor(get_query_emb(batch_sentences, checkpoint, batch_size), dtype=torch.float32)
        embeddings_list.append(embeddings)
    
    combined_embeddings = torch.cat(embeddings_list, dim=0)
    return combined_embeddings

class MyDataset(Dataset):
    def __init__(self, offers: List[str], true_matches: List[str], false_matches: List[str], checkpoint: Checkpoint, batch_size: int, batch_size2: int):
        self.offers = offers
        self.true_matches = true_matches
        self.false_matches = false_matches
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.batch_size2 = batch_size2

    def __len__(self):
        return len(self.offers)

    def __getitem__(self, idx):
        offer_embs = get_query_emb_batch([self.offers[idx]], self.checkpoint, batch_size=self.batch_size, batch_size2=self.batch_size2)
        true_match_embs = get_query_emb_batch([self.true_matches[idx]], self.checkpoint, batch_size=self.batch_size, batch_size2=self.batch_size2)
        false_match_embs = get_query_emb_batch([self.false_matches[idx]], self.checkpoint, batch_size=self.batch_size, batch_size2=self.batch_size2)

        y_true = torch.ones(len(true_match_embs))
        y_false = torch.zeros(len(false_match_embs))

        X = torch.cat([offer_embs, offer_embs], dim=0)
        X_pair = torch.cat([true_match_embs, false_match_embs], dim=0)
        y = torch.cat([y_true, y_false], dim=0)

        return X, X_pair, y

def vae_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def validate(vae, siamese, dataloader, device):
    vae.eval()
    siamese.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    mean_vae_time = 0
    mean_seamese_time = 0

    all_predictions = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for f, g, similarity in dataloader:
            f, g, similarity = f.to(device).reshape(-1, 32, 768), g.to(device).reshape(-1, 32, 768), similarity.to(device).reshape(-1)

            recon_f, mu_f, log_var_f = vae(f)
            recon_g, mu_g, log_var_g = vae(g)

            vae_time_start = time.time()
            loss_vae_f = vae_loss(recon_f, f, mu_f, log_var_f)
            loss_vae_g = vae_loss(recon_g, g, mu_g, log_var_g)
            mean_vae_time += time.time() - vae_time_start

            siamese_time_start = time.time()
            similarity_score = siamese(vae.reparameterize(mu_f, log_var_f), vae.reparameterize(mu_g, log_var_g))
            mean_seamese_time += time.time() - siamese_time_start

            loss_siamese = criterion(similarity_score, similarity.view(-1))

            loss = loss_vae_f + loss_vae_g + loss_siamese

            total_loss += loss.item()

            predicted_labels = (similarity_score > 0).float()
            correct_predictions += (predicted_labels == similarity.view(-1)).sum().item()
            total_samples += similarity.size(0)

            all_predictions.extend(similarity_score.detach().cpu().numpy())
            all_targets.extend(similarity.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    roc_auc = roc_auc_score(all_targets, all_predictions)
    mean_vae_time = mean_vae_time / total_samples
    mean_seamese_time = mean_seamese_time / total_samples

    return average_loss, accuracy, roc_auc, mean_vae_time, mean_seamese_time

def save_model(vae, siamese, optimizer, epoch, path='./models'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epoch,
        'vae_state_dict': vae.state_dict(),
        'siamese_state_dict': siamese.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))

def train(vae, siamese, train_dataloader, val_dataloader, lr, epochs=10, device='cuda'):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    with wandb.init(project="vae_siam", entity="igor-sondors"):
        wandb.config.update({
            "initial_learning_rate": lr,
            "epochs": epochs,
        })

        n_batches = len(train_dataloader)

        params = list(vae.parameters()) + list(siamese.parameters())
        optimizer = AdamW(params, lr=lr, eps=1e-8)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # Scheduler definition
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs*n_batches)

        scaler = torch.cuda.amp.GradScaler()
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            print(f"{epoch+1}/{epochs}")
            vae.train()
            siamese.train()
            total_train_loss = 0

            batch_time = time.time()
            for batch_index, (f, g, similarity) in enumerate(train_dataloader):
                f, g, similarity = f.to(device).reshape(-1, 32, 768), g.to(device).reshape(-1, 32, 768), similarity.to(device).reshape(-1)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    recon_f, mu_f, log_var_f = vae(f)
                    recon_g, mu_g, log_var_g = vae(g)

                    loss_vae_f = vae_loss(recon_f, f, mu_f, log_var_f)
                    loss_vae_g = vae_loss(recon_g, g, mu_g, log_var_g)

                    similarity_score = siamese(vae.reparameterize(mu_f, log_var_f), vae.reparameterize(mu_g, log_var_g))
                    loss_siamese = criterion(similarity_score, similarity.view(-1))

                    total_loss = loss_vae_f + loss_vae_g + loss_siamese

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += total_loss.item()
                print(f"{batch_index+1}/{n_batches}, batches_time = {round(time.time() - batch_time, 2)}")

            average_train_loss = total_train_loss / len(train_dataloader)
            average_val_loss, accuracy, roc_auc, mean_vae_time, mean_seamese_time = validate(vae, siamese, val_dataloader, device)

            print(f'Epoch {epoch+1}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}, Val ROCAUC: {roc_auc}, Time VAE|Siamese: {mean_vae_time}|{mean_seamese_time}')
            
            # Логирование метрик в wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': average_train_loss,
                'val_loss': average_val_loss,
                'val_accuracy': accuracy,
                'val_roc_auc': roc_auc,
                'val_mean_vae_time': mean_vae_time,
                'val_mean_siamese_time': mean_seamese_time,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            scheduler.step()  # Scheduler step
            # if (epoch + 1) % 2 == 0:
            save_model(vae, siamese, optimizer, epoch + 1)

    return vae, siamese

def load_data(pth_models, pth_offers, checkpoint, batch_size, batch_size2):
    id_category = {
        2801: 'мобильные телефоны'
    }
    
    df_models = pd.read_csv(pth_models, sep=';')
    df_offers = pd.read_csv(pth_offers, sep=';')

    df_models = df_models[df_models['category_id'].isin(id_category.keys())].reset_index(drop=True)
    df_offers = df_offers[df_offers['category_id'].isin(id_category.keys())].reset_index(drop=True)

    df_offers_shuffled = df_offers.sample(frac=1, random_state=42)

    test_size = int(0.15 * len(df_offers_shuffled))

    df_train = df_offers_shuffled.iloc[:-test_size]
    df_test = df_offers_shuffled.iloc[-test_size:]

    print("Размер тренировочной выборки:", len(df_train))
    print("Размер тестовой выборки:", len(df_test))

    offer_batch = list(df_train['name'])
    true_match_batch = list(df_train['true_match'])
    false_match_batch = list(df_train['false_match'])
    train_dataset = MyDataset(offer_batch, true_match_batch, false_match_batch, checkpoint, batch_size, batch_size2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    offer_batch = list(df_test['name'])
    true_match_batch = list(df_test['true_match'])
    false_match_batch = list(df_test['false_match'])
    test_dataset = MyDataset(offer_batch, true_match_batch, false_match_batch, checkpoint, batch_size, batch_size2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    ckpt_pth = "/home/sondors/Documents/ColBERT_weights/2801_lr04_bsize_210_apple/none/2024-04/18/09.16.10/checkpoints/colbert-187-finish"
    experiment = "colbert-187-finish"

    doc_maxlen = 300
    nbits = 2
    nranks = 1
    kmeans_niters = 4

    device = "cuda"
    checkpoint = load_model(ckpt_pth, doc_maxlen, nbits, kmeans_niters, device)

    pth_models = "/home/sondors/Documents/price/BERT_data/data/17-04-2024_Timofey/2801_offers_models_Apple.csv"
    pth_offers = '/home/sondors/Documents/price/BERT_data/data/17-04-2024_Timofey/2801_Apple_triplets_offer_model_train.csv'

    # Batch size parameters
    batch_size = 1500
    batch_size2 = 100000#3000

    epochs = 15
    initial_lr = 0.01

    train_dataloader, test_dataloader = load_data(pth_models, pth_offers, checkpoint, batch_size, batch_size2)

    model_vae = VAE().to(device)
    model_siamese = SiameseNetwork().to(device)

    vae, siamese = train(model_vae, model_siamese, train_dataloader, test_dataloader, initial_lr, epochs=epochs, device=device)
