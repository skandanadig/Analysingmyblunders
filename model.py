import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessCNNFeatureExtractor(nn.Module):
    """
    Shared 2D Convolutional Network to extract spatial features from an 8x8 chess board.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Input channels: 12 (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling to reduce spatial dimensions (8x8 -> 4x4)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Flattened size after pool: 128 channels * 4 * 4 spatial size = 2048
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)

    def forward(self, x):
        # x shape: (N, 12, 8, 8) where N = batch_size * sequence_length
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # Shape: (N, 64, 4, 4)
        x = F.relu(self.conv3(x))             # Shape: (N, 128, 4, 4)
        
        x = x.view(x.size(0), -1)             # Flatten to (N, 2048)
        x = F.relu(self.fc(x))                # Embed to (N, embed_dim)
        return x

class SpatioTemporalChessModel(nn.Module):
    """
    Hybrid CNN-LSTM architecture for processing a sequence of chess board states.
    """
    def __init__(self, embed_dim=256, lstm_hidden_dim=128, num_lstm_layers=1):
        super().__init__()
        
        # Step 1: Spatial Board Feature Extractor
        self.cnn = ChessCNNFeatureExtractor(embed_dim=embed_dim)
        
        # Step 2: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        """
        Forward pass for a sequence of boards.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, 12, 8, 8).
            For 20 consecutive moves, seq_len = 20.
            
        Returns
        -------
        final_hidden : torch.Tensor
            The final concatenated hidden state of the Bi-LSTM.
            Shape: (batch_size, lstm_hidden_dim * 2)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Step 1: Collapse the batch and sequence dimensions for the CNN
        # PyTorch handles this efficiently, acting identically to Keras' TimeDistributed
        # Shape becomes (batch_size * seq_len, 12, 8, 8)
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        
        # Extract spatial features for each frame independently
        spatial_embeddings = self.cnn(x_reshaped)
        
        # Reshape back to sequence format for the LSTM
        # Shape becomes (batch_size, seq_len, embed_dim)
        lstm_input = spatial_embeddings.view(batch_size, seq_len, -1)
        
        # Step 2: Pass the sequence through the Bi-LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        # h_n shape: (num_layers * 2, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Extract the final hidden state. 
        # For a Bi-LSTM, h_n has sets of features for both forward and backward directions.
        # We concatenate the final forward hidden state and final backward hidden state.
        final_forward = h_n[-2, :, :]  # Last forward state
        final_backward = h_n[-1, :, :] # Last backward state
        
        final_hidden = torch.cat((final_forward, final_backward), dim=1)
        # final_hidden shape: (batch_size, lstm_hidden_dim * 2)
        
        return final_hidden

if __name__ == "__main__":
    # Test the model with dummy data
    batch_size = 4
    seq_len = 20
    
    # Create dummy tensor mimicking 4 batches of 20 consecutive moves
    # Shape: (4, 20, 12, 8, 8)
    dummy_input = torch.randn(batch_size, seq_len, 12, 8, 8)
    
    model = SpatioTemporalChessModel(embed_dim=256, lstm_hidden_dim=128)
    
    output = model(dummy_input)
    
    print("--- Spatio-Temporal CNN-LSTM Model Test ---")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (batch_size, lstm_hidden_dim * 2)")
    print("Forward pass successful!")
