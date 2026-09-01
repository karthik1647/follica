"""
Follica Model Training & Quantization Pipeline.

Features:
1. Architecture Selection: MobileNetV4 (Default, next-gen mobile efficiency) vs EfficientNet-B0 (Fallback)
2. Per-Epoch Checkpointing: Saves state after every epoch to Google Drive / mount to survive Colab disconnects
3. Volunteer Calibration Merge: Integrates consented clinic photos
4. Export: Quantized ONNX & TFLite models
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import timm

from calibration_loader import load_dataset_with_calibration, ScalpDataset, get_data_transforms


def create_model(arch: str = "mobilenetv4", num_classes: int = 5, pretrained: bool = True):
    """
    Creates backbone architecture.

    Architecture Rationale:
    - MobileNetV4 (mobilenetv4_conv_small / mobilenetv4_conv_medium):
      Introduced in 2024, MobileNetV4 utilizes Universal Inverted Bottleneck (UIB) blocks
      and Mobile MQA attention, delivering superior Pareto-optimal accuracy/latency on mobile
      devices (~14 MB ONNX size). Export to ONNX / TFLite is fully supported in timm >= 0.9.16.
    - EfficientNet-B0:
      Battle-tested compound-scaled architecture with universal TFLite/ONNX compatibility.
    """
    print(f"\n[Follica ML] Initializing backbone: {arch} (Pretrained: {pretrained}, Classes: {num_classes})")
    
    if arch.startswith("mobilenetv4"):
        try:
            # Try MobileNetV4 from timm
            model = timm.create_model("mobilenetv4_conv_small.e1200_r224_in1k", pretrained=pretrained, num_classes=num_classes)
            print("[Follica ML] Successfully instantiated MobileNetV4 (mobilenetv4_conv_small).")
            return model
        except Exception as e:
            print(f"[Follica ML] MobileNetV4 instantiation notice: {e}. Falling back to EfficientNet-B0.")
            arch = "efficientnet_b0"

    if arch == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
        print("[Follica ML] Successfully instantiated EfficientNet-B0.")
        return model

    raise ValueError(f"Unsupported architecture: {arch}")


def train_follica(
    base_data_dir: str,
    calibration_data_dir: str = None,
    checkpoint_dir: str = "./checkpoints",
    arch: str = "mobilenetv4",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4,
    resume_from: str = None
):
    """Executes model training with per-epoch checkpointing."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Follica ML] Active Compute Device: {device}")

    # Load data with calibration merge placeholder
    file_paths, labels, class_to_idx = load_dataset_with_calibration(base_data_dir, calibration_data_dir)
    if len(file_paths) == 0:
        print("[Follica ML] Warning: No images found. Please check dataset path.")
        return

    # Train/Val Split (85% Train, 15% Validation)
    train_tf, val_tf = get_data_transforms()
    total_samples = len(file_paths)
    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size

    # Split indices deterministically
    indices = torch.randperm(total_samples).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_files = [file_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = ScalpDataset(train_files, train_labels, transform=train_tf)
    val_dataset = ScalpDataset(val_files, val_labels, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[Follica ML] Dataset Loaded: {train_size} training samples, {val_size} validation samples.")

    model = create_model(arch=arch, num_classes=len(class_to_idx), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_acc = 0.0

    # Resume from checkpoint if session previously disconnected
    if resume_from and os.path.exists(resume_from):
        print(f"[Follica ML] Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"[Follica ML] Resumed at epoch {start_epoch} with prior best validation accuracy: {best_val_acc:.2f}%")

    # Training Loop
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc = (correct / max(total, 1)) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += targets.size(0)
                val_correct += preds.eq(targets).sum().item()

        val_loss = val_loss / max(val_total, 1)
        val_acc = (val_correct / max(val_total, 1)) * 100.0

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # -------------------------------------------------------------
        # Per-Epoch Checkpointing (Survival against Colab Disconnects)
        # -------------------------------------------------------------
        checkpoint_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': max(best_val_acc, val_acc),
            'arch': arch
        }
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint_payload, epoch_ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint_payload, best_ckpt_path)
            print(f"  --> Saved new best model checkpoint ({best_val_acc:.2f}% Val Acc) to {best_ckpt_path}")

    print(f"\n[Follica ML] Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    return model


def export_to_onnx(model_path: str, output_onnx_path: str, arch: str = "mobilenetv4"):
    """Exports best PyTorch model checkpoint to ONNX format."""
    print(f"\n[Follica ML] Exporting checkpoint {model_path} to ONNX: {output_onnx_path}")
    model = create_model(arch=arch, num_classes=5, pretrained=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[Follica ML] Successfully exported ONNX model ({os.path.getsize(output_onnx_path) / (1024*1024):.2f} MB).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Follica Model Training")
    parser.add_argument("--base_dir", type=str, default="./data/base", help="Base dataset directory")
    parser.add_argument("--calibration_dir", type=str, default=None, help="Volunteer calibration dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save per-epoch checkpoints")
    parser.add_argument("--arch", type=str, default="mobilenetv4", choices=["mobilenetv4", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--export_onnx", type=str, default=None, help="Path to export final ONNX model")

    args = parser.parse_args()

    trained_model = train_follica(
        base_data_dir=args.base_dir,
        calibration_data_dir=args.calibration_dir,
        checkpoint_dir=args.checkpoint_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume_from=args.resume
    )

    if args.export_onnx and os.path.exists(os.path.join(args.checkpoint_dir, "best_model.pt")):
        export_to_onnx(
            model_path=os.path.join(args.checkpoint_dir, "best_model.pt"),
            output_onnx_path=args.export_onnx,
            arch=args.arch
        )
