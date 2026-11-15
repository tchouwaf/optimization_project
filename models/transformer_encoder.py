import torch
import torch.nn as nn


class IMDbTransformer(nn.Module):

    def __init__(self, vocab_size=30522, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, max_length=256, dropout=0.1):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        
        input_ids = X[:, 0, :].long()
        attention_mask = X[:, 1, :].long()

        batch_size, seq_length = input_ids.shape

        positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)

        x = self.embedding(input_ids)
        x = x + self.pos_embedding(positions)

        x = self.dropout(x)

        padding_mask = attention_mask == 0

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x[:, 0, :]

        return self.classifier(x)