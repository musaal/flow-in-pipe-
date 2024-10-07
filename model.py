import torch.nn as nn
import torch

class Transformer(nn.Module):
    def __init__(self, feature_size=6 , num_layers=3, dropout=0.1):
        super(Transformer, self).__init__()

        # Using batch_first=True to optimize the model performance
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=3, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Decoder layer to predict the next step
        self.decoder = nn.Linear(feature_size, 6)  # Change output to match target size (8)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz, device):
        # Creating the mask more efficiently by splitting into manageable sizes
        mask = torch.triu(torch.ones(sz, sz, device=device), 1)  # Shift above diagonal
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, device):
        # Mask for the sequence to handle auto-regression
        mask = self._generate_square_subsequent_mask(src.size(1), device)

        # Pass through transformer encoder
        output = self.transformer_encoder(src, mask=mask)

        # Decode to predict the next feature values (8 outputs)
        output = self.decoder(output)

        # If you want to get the final output for the batch (e.g., last time step)
        # output = output[:, -1, :]  # Uncomment if you want only the last output
        return output
