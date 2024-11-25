
import torch
from torch import nn

class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size, hidden_size):
        super(NeuMF, self).__init__()

        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_size)

        # Movie embedding
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # GMF layer
        self.gmf = nn.Linear(embedding_size, 1)

    def forward(self, user_input, movie_input, max_rating, min_rating):
        # User and movie embeddings
        user_embed = self.user_embedding(user_input)
        movie_embed = self.movie_embedding(movie_input)

        # Element-wise product for GMF
        gmf_output = torch.mul(user_embed, movie_embed)

        # Concatenate user and movie embeddings for MLP
        mlp_input = torch.cat([user_embed, movie_embed], dim=1)

        # MLP forward pass
        mlp_output = self.mlp(mlp_input)
        # Concatenate GMF and MLP outputs
        combined_output = torch.cat([gmf_output, mlp_output], dim=1)

        raw_prediction = torch.sum(combined_output, dim=1)
        prediction = torch.sigmoid(raw_prediction)

        # Reverse the normalization to get the prediction in the original scale
        original_scale_prediction = prediction * (max_rating - min_rating) + min_rating

        return original_scale_prediction


