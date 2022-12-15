import argparse
import logging

import pandas as pd
import numpy as np

import torch
from torchmetrics import RetrievalMAP, RetrievalNormalizedDCG
from torch import nn
from torchtext.vocab import vocab
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from ncf_torch import NCF

class MovieLensDataset(Dataset):
    def __init__(self, df, user_vocab, movie_vocab):
        self.user_ids = df.user_id.astype(str).to_numpy()
        self.movie_titles = df.movie_title.astype(str).to_numpy()
        self.ratings = df.rating.astype(np.float32).to_numpy()

        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return self.user_vocab[self.user_ids[idx]], self.movie_vocab[self.movie_titles[idx]], self.ratings[idx]

def train_loop(model, criterion, optimizer, train_loader, test_loader, scorer, device):
    model.train()
    total_loss = []
    for (users, movies, ratings) in train_loader:
        (users, movies, ratings) = users.to(device), movies.to(device), ratings.to(device)
        optimizer.zero_grad()
        out = model(users, movies)
        loss = criterion(out, ratings)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    total_loss = np.mean(total_loss)

    model.eval()
    valid_loss = []
    valid_score = []
    preds, trues, indexes = [], [], []
    with torch.no_grad():
        for (users, movies, ratings) in test_loader:
            (users, movies, ratings) = users.to(device), movies.to(device), ratings.to(device)
            out = model(users, movies)
            loss = criterion(out, ratings)
            valid_loss.append(loss.item())
            preds.append(out)
            trues.append(ratings)
            indexes.append(users)
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    indexes = torch.cat(indexes)
    valid_loss = np.mean(valid_loss)
    valid_score = scorer(preds, trues, indexes=indexes).item()

    return total_loss, valid_loss, valid_score

def log_metrics(epoch, train_loss, valid_loss, valid_score):
    print(f"epoch {epoch}:")
    print(f"train_loss={train_loss}")
    print(f"valid_loss={valid_loss}")
    print(f"ndcg={valid_score}")
    print()
    logging.info(f"epoch {epoch}: train_loss={train_loss}, valid_loss={valid_loss}, ndcg={valid_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="movie recommendation model training")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--num_mlp_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("reading data")
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")
    
    users = train.user_id.astype(str).to_numpy()
    movies = train.movie_title.astype(str).to_numpy()
    user_vocab = vocab(Counter(users), specials=['OOV'])
    user_vocab.set_default_index(0)
    movie_vocab = vocab(Counter(movies), specials=['OOV'])
    movie_vocab.set_default_index(0)

    batch_size = args.batch_size
    train_data = MovieLensDataset(train, user_vocab, movie_vocab)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = MovieLensDataset(test, user_vocab, movie_vocab)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    embedding_dimensions = args.embedding_dim
    dense_dims = [embedding_dimensions for _ in range(args.num_mlp_layers)]
    model = NCF(users, movies, embedding_dimension=embedding_dimensions, dense_dims=dense_dims)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    logging.info("starting training")
    scorer = RetrievalNormalizedDCG(k = 10)
    epochs = args.epochs
    for epoch in range(1, epochs+1):
        train_loss, valid_loss, score = train_loop(model, criterion, optimizer, train_loader, test_loader, scorer, device)
        log_metrics(epoch, train_loss, valid_loss, score)