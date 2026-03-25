"""
PostureLSTM — LSTM-based Temporal Posture Classifier
====================================================
THIS is the key differentiator — temporal modeling.

Why LSTM?
  Posture is NOT a single-frame problem. A person might
  momentarily lean forward (reaching for something) — that's
  NOT bad posture. But sustained forward lean over 3 seconds
  IS slouching. LSTM captures this temporal pattern.

Architecture:
  Input: (batch, seq_len=30, features=12)
       → LSTM(hidden=64, layers=2, bidirectional)
       → Attention pooling over time steps
       → Linear(128) → ReLU → Dropout(0.3)
       → Linear(5 classes)

The attention mechanism lets the model focus on the most
informative frames in the window, rather than just using
the last hidden state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Attention mechanism over LSTM time steps.
    Learns which frames in the window matter most.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted sum
        """
        scores = self.attention(lstm_output)          # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)            # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context


class PostureLSTM(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 5,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.directions

        # Attention pooling
        self.attention = TemporalAttention(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) — temporal window of features
        Returns:
            logits: (batch, num_classes)
        """
        # Normalize input features
        x = self.input_norm(x)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * directions)

        # Attention-weighted pooling over time steps
        context = self.attention(lstm_out)
        # context: (batch, hidden_size * directions)

        # Classify
        logits = self.classifier(context)
        return logits

    def predict(self, x: torch.Tensor) -> tuple:
        """Convenience: returns (class_indices, probabilities)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with long-term memory)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PostureLSTM(input_size=12, hidden_size=64, num_layers=2, num_classes=5)
    print(f"LSTM Parameters: {model.count_parameters():,}")
    print(model)

    x = torch.randn(4, 30, 12)   # batch=4, seq_len=30, features=12
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Probs shape: {probs.shape}")
