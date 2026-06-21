import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

# Import the base model we created in Prompt 4
from model import SpatioTemporalChessModel

class BlunderMetaLearner(nn.Module):
    """
    Final Meta-Learner Head that fuses spatial-temporal board embeddings 
    with external engine metrics to predict tactical blunders.
    """
    def __init__(self, cnn_embed_dim=256, lstm_hidden_dim=128, num_external_metrics=8):
        """
        num_external_metrics: The size of our secondary feature vector 
        (e.g., 7-dim from Prompt 3 + 1 for centipawn delta).
        """
        super().__init__()
        
        # 1. Base spatio-temporal model (Bi-LSTM outputs size: lstm_hidden_dim * 2)
        self.base_model = SpatioTemporalChessModel(
            embed_dim=cnn_embed_dim, 
            lstm_hidden_dim=lstm_hidden_dim
        )
        
        # Input to the meta-head is the concatenated size of LSTM output + external metrics
        meta_input_dim = (lstm_hidden_dim * 2) + num_external_metrics
        
        # 2. Shallow fully connected network
        self.fc1 = nn.Linear(meta_input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, board_seq, external_metrics):
        """
        board_seq: shape (batch_size, 20, 12, 8, 8)
        external_metrics: shape (batch_size, num_external_metrics)
        """
        # Get hidden state from Bi-LSTM -> shape: (batch_size, lstm_hidden_dim * 2)
        lstm_hidden = self.base_model(board_seq)
        
        # Concatenate Bi-LSTM state with the external engine/time-pressure metrics
        # Shape becomes (batch_size, lstm_hidden_dim * 2 + num_external_metrics)
        combined_features = torch.cat((lstm_hidden, external_metrics), dim=1)
        
        # Pass through shallow Meta-Learner network
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        out = self.fc2(x)
        
        # Sigmoid activation to predict probability [0, 1]
        prob = self.sigmoid(out) 
        
        return prob.squeeze(-1) # Return shape (batch_size,)


def train_meta_learner(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 5,
    device: str = "cpu"
):
    """
    Training loop for the Blunder Meta-Learner featuring Early Stopping,
    Binary Cross-Entropy Loss, and the Adam Optimizer.
    """
    model = model.to(device)
    
    # Loss function & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # State tracking for Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print(f"Starting training on device: {device}")
    
    for epoch in range(num_epochs):
        # ---------------------
        #    Training Phase
        # ---------------------
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # We expect the dataloader to yield dicts with:
            # "board_seq" (B, 20, 12, 8, 8), "metrics" (B, N), and "target" (B,)
            # Target generation logic (delta > 2.0 -> 1.0) is handled in the Dataset.
            boards = batch["board_seq"].to(device)
            metrics = batch["metrics"].to(device)
            targets = batch["target"].to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(boards, metrics)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass & Optimizer step
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * boards.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # ---------------------
        #   Validation Phase
        # ---------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                boards = batch["board_seq"].to(device)
                metrics = batch["metrics"].to(device)
                targets = batch["target"].to(device)
                
                predictions = model(boards, metrics)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item() * boards.size(0)
                
                # Calculate accuracy using 0.5 as blunder threshold
                preds_binary = (predictions >= 0.5).float()
                correct += (preds_binary == targets).sum().item()
                total += targets.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch+1:>2}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # ---------------------
        #    Early Stopping
        # ---------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best weights
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"-> Early stopping triggered after {epoch+1} epochs! (Best Val Loss: {best_val_loss:.4f})")
                break
                
    # Load the best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored model weights from the best validation epoch.")
        
    return model

# ==============================================================================
# Example execution using dummy data
# ==============================================================================
if __name__ == "__main__":
    print("Initializing Meta-Learner Test...")
    
    # Generate dummy data for a batch of 8
    batch_size, seq_len, num_metrics = 8, 20, 8
    dummy_boards = torch.randn(batch_size, seq_len, 12, 8, 8)
    dummy_metrics = torch.randn(batch_size, num_metrics)
    
    # Binary targets representing whether the move evaluated to a drop > 2.0
    dummy_targets = torch.randint(0, 2, (batch_size,)).float()
    
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self): return batch_size
        def __getitem__(self, idx):
            return {
                "board_seq": dummy_boards[idx],
                "metrics": dummy_metrics[idx],
                "target": dummy_targets[idx]
            }
            
    loader = DataLoader(MockDataset(), batch_size=4, shuffle=True)
    
    meta_model = BlunderMetaLearner(num_external_metrics=num_metrics)
    
    # Run the training loop for a maximum of 5 epochs to demonstrate functionality
    train_meta_learner(meta_model, train_loader=loader, val_loader=loader, num_epochs=5, patience=2)
