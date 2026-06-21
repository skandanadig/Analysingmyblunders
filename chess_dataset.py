import torch
import chess
from torch.utils.data import Dataset
from typing import List, Dict, Any

class ChessDataset(Dataset):
    """
    Custom PyTorch Dataset for Chess board states.
    
    Each item in the dataset returns:
      - board: Tensor of shape (12, 8, 8) representing 6 piece types x 2 colors.
      - meta:  Tensor of shape (7,) representing clock time, move number, 
               castling rights, and active turn.
      - target: Optional tensor representing a label (e.g. evaluation delta).
      
    Expects data_samples to be a list of dictionaries:
      [
          {
              "board": chess.Board(...),
              "clock_time": 180.5, # Time remaining in seconds
              "move_number": 15,   # Current move number
              "target": 0.45       # Target prediction value
          },
          ...
      ]
    """
    def __init__(self, data_samples: List[Dict[str, Any]]):
        self.data_samples = data_samples

    def __len__(self):
        return len(self.data_samples)

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Converts a chess.Board into a 12x8x8 one-hot encoded tensor."""
        # 12 channels (6 piece types * 2 colors), 8 ranks, 8 files
        tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Color offset: White is 0, Black is 6
                color_offset = 0 if piece.color == chess.WHITE else 6
                
                # Piece types in python-chess are 1-indexed (PAWN=1 to KING=6)
                # We subtract 1 to make it 0-indexed (0 to 5)
                piece_idx = piece.piece_type - 1
                
                channel = color_offset + piece_idx
                
                # Rank (0-7 representing 1-8) and File (0-7 representing A-H)
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                tensor[channel, rank, file] = 1.0
                
        return tensor

    def _get_metadata(self, board: chess.Board, clock_time: float, move_number: int) -> torch.Tensor:
        """Extracts secondary metadata into a 1D numeric feature vector."""
        # Castling rights (1.0 if available, 0.0 if not)
        wk = float(board.has_kingside_castling_rights(chess.WHITE))
        wq = float(board.has_queenside_castling_rights(chess.WHITE))
        bk = float(board.has_kingside_castling_rights(chess.BLACK))
        bq = float(board.has_queenside_castling_rights(chess.BLACK))
        
        # Turn: 1.0 for White, 0.0 for Black
        turn = 1.0 if board.turn == chess.WHITE else 0.0
        
        # Combine into a single numeric feature vector of size 7
        meta = torch.tensor([
            float(clock_time),
            float(move_number),
            wk,
            wq,
            bk,
            bq,
            turn
        ], dtype=torch.float32)
        
        return meta

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data_samples[idx]
        board = sample["board"]
        
        # Default clock to 0.0 if not explicitly provided
        clock_time = sample.get("clock_time", 0.0)
        # Default move number to the board's internal fullmove number if not provided
        move_number = sample.get("move_number", board.fullmove_number)
        
        board_tensor = self._board_to_tensor(board)
        meta_tensor = self._get_metadata(board, clock_time, move_number)
        
        result = {
            "board": board_tensor,
            "meta": meta_tensor,
        }
        
        # Include target (like delta) if available in the dictionary
        if "target" in sample:
            result["target"] = torch.tensor(sample["target"], dtype=torch.float32)
            
        return result

# ---------------------------------------------------------------------------
# Example usage (can be tested directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy board state and make a move
    b = chess.Board()
    b.push_san("e4")
    
    # Create sample data
    samples = [
        {
            "board": b,
            "clock_time": 600.0, # 10 minutes left
            "move_number": b.fullmove_number,
            "target": 0.35       # Dummy engine eval delta
        }
    ]
    
    # Instantiate dataset
    dataset = ChessDataset(samples)
    
    # Fetch the first item from the dataset
    item = dataset[0]
    
    print("--- PyTorch Dataset Test ---")
    print(f"Board tensor shape: {item['board'].shape}")
    print(f"Meta tensor shape:  {item['meta'].shape}")
    print(f"Target tensor:      {item['target']}\n")
    
    # Print the metadata values explicitly
    print("Metadata Vector Content:")
    meta_labels = ["Clock Time", "Move Num", "W_Kingside", "W_Queenside", "B_Kingside", "B_Queenside", "Turn (White=1)"]
    for label, val in zip(meta_labels, item['meta']):
        print(f"  {label:<15}: {val.item()}")
