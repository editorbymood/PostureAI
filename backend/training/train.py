"""
PostureAI Training Pipeline
============================
Trains both MLP (baseline) and LSTM (temporal) classifiers.

Usage:
    python -m backend.training.train --model lstm --epochs 50
    python -m backend.training.train --model mlp --epochs 30

Features:
  - Early stopping with patience
  - Learning rate scheduling
  - Train/val split with stratification
  - Model checkpointing
  - Training metrics logging
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mlp_classifier import PostureMLP
from models.lstm_classifier import PostureLSTM
from models.feature_eng import NUM_FEATURES, NUM_CLASSES, POSTURE_LABELS
from training.dataset import PostureDataset, load_all_data, generate_synthetic_data
from training.evaluate import evaluate_model, print_report


def train_mlp(args):
    """Train MLP classifier on single-frame features."""
    print("\n" + "="*60)
    print("  Training PostureMLP (Single-Frame Baseline)")
    print("="*60)

    # Load data
    X, y = load_all_data(args.data_dir, temporal=False)
    if X is None:
        print("⚠️  No training data found. Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=2000, temporal=False)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {NUM_CLASSES} classes")
    print(f"Class distribution: {np.bincount(y, minlength=NUM_CLASSES)}")

    # Train/val split (80/20)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long))
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = PostureMLP(input_size=NUM_FEATURES, num_classes=NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device} | Parameters: {model.count_parameters():,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        train_loss /= train_total
        val_loss /= max(val_total, 1)
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / max(val_total, 1) * 100
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.model_dir, 'posture_mlp.pth')
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Final evaluation ──
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'posture_mlp.pth')))
    print("\n📊 Final Evaluation:")
    evaluate_model(model, val_loader, device, POSTURE_LABELS)
    print(f"\n✅ Model saved: {os.path.join(args.model_dir, 'posture_mlp.pth')}")


def train_lstm(args):
    """Train LSTM classifier on temporal feature windows."""
    print("\n" + "="*60)
    print("  Training PostureLSTM (Temporal Model)")
    print("="*60)

    # Load data
    X, y = load_all_data(args.data_dir, temporal=True, window_size=args.window_size)
    if X is None:
        print("⚠️  No training data found. Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=2000, temporal=True, window_size=args.window_size)

    print(f"Dataset: {X.shape[0]} windows, seq_len={X.shape[1]}, features={X.shape[2]}")
    print(f"Class distribution: {np.bincount(y, minlength=NUM_CLASSES)}")

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.long))
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = PostureLSTM(input_size=NUM_FEATURES, hidden_size=64,
                        num_layers=2, num_classes=NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device} | Parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (out.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        train_loss /= train_total
        val_loss /= max(val_total, 1)
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / max(val_total, 1) * 100
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.model_dir, 'posture_lstm.pth')
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'posture_lstm.pth')))
    print("\n📊 Final Evaluation:")
    evaluate_model(model, val_loader, device, POSTURE_LABELS, is_temporal=True)
    print(f"\n✅ Model saved: {os.path.join(args.model_dir, 'posture_lstm.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PostureAI classifiers")
    parser.add_argument('--model', choices=['mlp', 'lstm', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent.parent.parent / 'data'))
    parser.add_argument('--model_dir', default=str(Path(__file__).resolve().parent.parent.parent / 'models'))
    args = parser.parse_args()

    if args.model in ('mlp', 'both'):
        train_mlp(args)
    if args.model in ('lstm', 'both'):
        train_lstm(args)
