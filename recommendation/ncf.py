import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from tqdm.notebook import tqdm
from model import NeuMF

from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split

from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_movie_data():
    movielens_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = urlopen(movielens_url)
    with ZipFile(BytesIO(response.read())) as z:
        z.extractall()


ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
user_ids = ratings_df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = ratings_df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

ratings_df["user"] = ratings_df["userId"].map(user2user_encoded)
ratings_df["movie"] = ratings_df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
ratings_df["rating"] = ratings_df["rating"].values.astype(float)

min_rating = min(ratings_df["rating"])
max_rating = max(ratings_df["rating"])

print("Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
    num_users, num_movies, min_rating, max_rating
))


df = ratings_df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

embedding_size = 50
hidden_sizes = 64


# Convert data to PyTorch tensors
train_data = TensorDataset(torch.tensor(x_train[:, 0]), torch.tensor(x_train[:, 1]), torch.tensor(y_train))
val_data = TensorDataset(torch.tensor(x_val[:, 0]), torch.tensor(x_val[:, 1]), torch.tensor(y_val))

# Define DataLoader for training and validation
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

neumf_model = NeuMF(num_users, num_movies, embedding_size, hidden_sizes)
neumf_model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(neumf_model.parameters(), lr=0.001)


num_epochs = 20
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(neumf_model.parameters(), lr=0.001)

checkpoint_path = '/content/neumf_model_checkpoint.pth'


for epoch in range(num_epochs):
    neumf_model.train()

    for user_batch, item_batch, label_batch in train_loader:
        # Move data to GPU
        #user_batch, item_batch, label_batch = user_batch.to('cuda'), item_batch.to('cuda'), label_batch.to('cuda')

        optimizer.zero_grad()

        # Forward pass
        predictions = neumf_model(user_batch, item_batch, max_rating,min_rating)
        loss = criterion(predictions, label_batch.float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation
    neumf_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for user_batch, item_batch, label_batch in val_loader:
            # Move data to GPU
            #user_batch, item_batch, label_batch = user_batch.to('cuda'), item_batch.to('cuda'), label_batch.to('cuda')

            # Forward pass
            predictions = neumf_model(user_batch, item_batch,max_rating,min_rating)
            val_loss += criterion(predictions, label_batch.float())
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    torch.save(neumf_model, 'final_neumf_model.pth')



model_path = 'final_neumf_model.pth'
model_state_dict = torch.load(model_path)

# Create an instance of the NeuMF model
model = NeuMF(num_users, num_movies, embedding_size, hidden_sizes)
model.eval()

# Example
for i in range (20):
  user_id = torch.tensor([i])
  movie_id = torch.tensor([15])

  # Make a prediction
  with torch.no_grad():
    prediction = model(user_id, movie_id, max_rating, min_rating)

  print(f'User {i} has a Predicted Rating: {prediction.item()}')