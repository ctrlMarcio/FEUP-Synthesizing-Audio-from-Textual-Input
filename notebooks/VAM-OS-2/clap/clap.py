import torch
import torch.nn.functional as F
import torch.nn as nn


class Clap(nn.Module):
    def __init__(self,
                 audio_feature_dim,
                 text_feature_dim,
                 shared_embedding_dim
                 ):
        super().__init__()
        
        # Initialize the weights that will connect the input audio and text
        # features to the shared embedding space.
        self._audio_projection = torch.nn.Linear(
            audio_feature_dim, shared_embedding_dim)
        self._text_projection = torch.nn.Linear(
            text_feature_dim, shared_embedding_dim)

        # Initialize the temperature parameter used for scaling the pairwise
        # cosine similarities between image and text embeddings.
        self._learned_temperature = torch.nn.Parameter(torch.tensor([1.0]))

    def encode_audio(self, audio_features):
        # Project the audio features into the shared embedding space and
        # normalize the resulting embedding vectors.
        audio_embeddings = F.normalize(
            self._audio_projection(audio_features), p=2, dim=1)
        return audio_embeddings

    def encode_text(self, text_features):
        # Project the text features into the shared embedding space and
        # normalize the resulting embedding vectors.
        text_embeddings = F.normalize(
            self._text_projection(text_features), p=2, dim=1)
        return text_embeddings

    def forward(self, audio_features, text_features):
        # Encode the audio and text features into their respective embedding
        # spaces.
        audio_embeddings = self.encode_audio(audio_features)
        text_embeddings = self.encode_text(text_features)

        # Compute the pairwise cosine similarities between the image and text
        # embeddings, scaled by the learned temperature parameter.
        pairwise_similarities = torch.matmul(
            audio_embeddings, text_embeddings.T) * torch.exp(
            self._learned_temperature)

        # Compute the symmetric cross-entropy loss between the predicted
        # pairwise similarities and the true pairwise similarities.
        labels = torch.arange(audio_features.size(0))
        loss_i = F.cross_entropy(
            pairwise_similarities, labels, reduction='mean')
        loss_t = F.cross_entropy(
            pairwise_similarities.T, labels, reduction='mean')
        loss = (loss_i + loss_t) / 2

        return loss
