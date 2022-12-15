import numpy as np
import torch
from torch import nn

class NCF(nn.Module):
    def __init__(
        self,
        user_ids,
        movie_titles,
        embedding_dimension=32,
        dense_dims=[128, 64, 32],
    ):
        super().__init__()

        self.unique_user_ids = np.unique(user_ids)
        self.unique_movie_titles = np.unique(movie_titles)

        self.user_embeddings_gmf = nn.Embedding(len(self.unique_user_ids) + 1, embedding_dimension)
        self.user_embeddings_mlp = nn.Embedding(len(self.unique_user_ids) + 1, embedding_dimension)
        self.movie_embeddings_gmf = nn.Embedding(len(self.unique_movie_titles) + 1, embedding_dimension)
        self.movie_embeddings_mlp = nn.Embedding(len(self.unique_movie_titles) + 1, embedding_dimension)
        
        net = [nn.Linear(embedding_dimension*2, dense_dims[0]), nn.ReLU()]
        for i in range(1, len(dense_dims)):
            net.append(nn.Linear(dense_dims[i-1], dense_dims[i]))
            net.append(nn.ReLU())
        self.mlp = nn.Sequential(*net)

        self.nmf = nn.Linear(dense_dims[-1] + embedding_dimension, 1)

    def forward(self, user_id, movie_title):

        user_embedding_gmf = self.user_embeddings_gmf(user_id)
        movie_embedding_gmf = self.movie_embeddings_gmf(movie_title)
        hidden_gmf = user_embedding_gmf * movie_embedding_gmf

        user_embedding_mlp = self.user_embeddings_mlp(user_id)
        movie_embedding_mlp = self.movie_embeddings_mlp(movie_title)
        hidden_mlp = self.mlp(torch.cat([user_embedding_mlp, movie_embedding_mlp], axis=-1))

        return self.nmf(torch.cat([hidden_gmf, hidden_mlp], axis=-1)).squeeze(-1)