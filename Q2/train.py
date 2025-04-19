import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import os
from ViTDiff import VisionTransformerDiff
from config import *
import numpy as np
from tqdm import tqdm, trange

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train_model(config, train_transform=None, test_transform=None):
    """
    Train and evaluate a ViT model with given configuration
    
    Args:
        config: Dictionary containing model and training configuration
        train_transform: Optional custom transform for training data
        test_transform: Optional custom transform for test data
    
    Returns:
        float: Test accuracy
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup transforms
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    
    # Load datasets
    trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize model
    model = VisionTransformerDiff(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=10,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        pos_embed_type=config.get('pos_embed_type', '1d_learned')  # Default to 1D learned
    ).to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch} [Train]', 
                    leave=False, ncols=100)
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (pbar.n + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
                
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(dataloader, phase="val"):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        class_correct = [0] * 10
        class_total = [0] * 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        
        pbar = tqdm(dataloader, desc=f'[{phase.capitalize()}]', 
                    leave=False, ncols=100)
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                
                # Only calculate loss during validation, not during test
                if phase == "val":
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate per-class accuracy
                c = (predicted == targets).squeeze()
                for i in range(len(targets)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                # Update progress bar
                current_acc = 100. * correct / total
                pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
        
        epoch_acc = 100. * correct / total
        
        # Calculate per-class accuracies
        class_accuracies = {
            classes[i]: 100 * class_correct[i] / class_total[i] 
            for i in range(10)
        }
        
        if phase == "val":
            epoch_loss = running_loss / len(dataloader)
            return epoch_loss, epoch_acc, class_accuracies
        else:
            return None, epoch_acc, class_accuracies
    
    # Training loop
    best_val_acc = 0
    epochs_pbar = trange(config['epochs'], desc='Training Progress', ncols=100)
    
    for epoch in epochs_pbar:
        # Training phase
        train_loss, train_acc = train_one_epoch(epoch)
        
        # Validation phase
        val_loss, val_acc, class_accuracies = evaluate(testloader, phase="val")
        
        scheduler.step()
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"class_acc_{cls}": acc for cls, acc in class_accuracies.items()}
        })
        
        # Update progress bar description
        epochs_pbar.set_postfix({
            'train_acc': f'{train_acc:.2f}%',
            'val_acc': f'{val_acc:.2f}%'
        })
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Create experiment-specific directory
            exp_save_dir = os.path.join(SAVE_DIR, config.get("experiment_name", "default"))
            os.makedirs(exp_save_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, os.path.join(exp_save_dir, f'best_model.pth'))
    
    # Load best model for final evaluation
    exp_save_dir = os.path.join(SAVE_DIR, config.get("experiment_name", "default"))
    best_checkpoint = torch.load(os.path.join(exp_save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final test evaluation
    _, test_acc, class_accuracies = evaluate(testloader, phase="test")
    
    print("\nFinal Test Results:")
    print(f'Test Acc: {test_acc:.2f}%')
    print("\nPer-class accuracies:")
    for cls, acc in class_accuracies.items():
        print(f'{cls}: {acc:.2f}%')
    
    return test_acc

if __name__ == "__main__":
    # If run directly, train with base configuration
    wandb.init(
        project="differential-vit-cifar-10",
        group="base_training",
        name="base_config",
        config=CONFIG,
        dir=BASE_DIR
    )
    
    test_acc = train_model(wandb.config)
    wandb.finish() 