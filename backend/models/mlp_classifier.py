"""
PostureMLP — Multi-Layer Perceptron Classifier
===============================================
Baseline model for single-frame posture classification.

Architecture:
  Input (12 features) → Linear(64) → BN → ReLU → Dropout(0.3)
                       → Linear(32) → BN → ReLU → Dropout(0.2)
                       → Linear(5 classes)

Use case:
  Quick inference, solid baseline accuracy.
  Compare against LSTM to show temporal gains.
"""

import torch
import torch.nn as nn


class PostureMLP(nn.Module):
    def __init__(self, input_size: int = 12, num_classes: int = 5,
                 hidden_sizes: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers = []
        prev_size = input_size

        for i, h_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout if i == 0 else dropout * 0.7),
            ])
            prev_size = h_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_size) — single frame features
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> tuple:
        """Convenience: returns (class_index, probabilities)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = PostureMLP(input_size=12, num_classes=5)
    print(f"MLP Parameters: {model.count_parameters():,}")
    print(model)

    # Test forward pass
    x = torch.randn(8, 12)  # batch of 8, 12 features
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs[0].numpy()}")
