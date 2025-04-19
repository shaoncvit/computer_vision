import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Define directories as constants
DATA_DIR = '/ssd_scratch/cvit/shaon/cv_5/data'
MODEL_SAVE_DIR = '/ssd_scratch/cvit/shaon/cv_5/Q1/models_vit'
WANDB_DIR = '/ssd_scratch/cvit/shaon/cv_5/Q1/wandb'

# Configure wandb to use custom directory
os.environ['WANDB_DIR'] = WANDB_DIR
os.makedirs(WANDB_DIR, exist_ok=True)

def init_wandb(experiment_name, config=None):
    """
    Initialize wandb with custom directory
    """
    try:
        wandb.init(
            project='vision-transformer-cifar10',
            name=experiment_name,
            config=config,
            dir=WANDB_DIR
        )
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")

def get_cifar10_dataloaders(batch_size=128, num_workers=4, augment=False, val_split=0.1, distributed=False, world_size=None, rank=None):
    """
    Get CIFAR-10 dataloaders with train/val/test split and distributed support
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        augment: Whether to use data augmentation
        val_split: Fraction of training data to use for validation
        distributed: Whether to use distributed training
        world_size: Total number of processes (GPUs)
        rank: Process rank
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the full training set
    full_trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train)
    
    # Calculate lengths for train and validation
    val_size = int(len(full_trainset) * val_split)
    train_size = len(full_trainset) - val_size
    
    # Split into train and validation sets
    trainset, valset = random_split(
        full_trainset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders with appropriate sampler for distributed training
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    # Pin memory for faster data transfer to GPU
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True
    )
    
    valloader = DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True
    )

    # Load the test set
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test)
    
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset, num_replicas=world_size, rank=rank)
    else:
        test_sampler = None
        
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler,
        persistent_workers=True
    )

    return trainloader, valloader, testloader

def setup_distributed(rank, world_size):
    """
    Setup distributed training
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_model(model, trainloader, valloader, testloader, num_epochs=50, lr=1e-3, experiment_name=None, distributed=False, rank=0):
    """
    Train the model with distributed support
    """
    # Initialize wandb only on the main process if distributed
    if (not distributed or rank == 0) and experiment_name:
        init_wandb(experiment_name, config={
            'learning_rate': lr,
            'epochs': num_epochs,
            'batch_size': trainloader.batch_size,
            'architecture': model.__class__.__name__
        })

    # Move model to GPU and wrap with DDP if using distributed training
    model = model.cuda()
    if distributed:
        model = DDP(model, device_ids=[rank])
    else:
        # Use DataParallel if not using DDP but multiple GPUs are available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': None,
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Only show progress bar on rank 0 if distributed
    if not distributed or rank == 0:
        epoch_pbar = tqdm(range(num_epochs), desc='Training epochs')
    else:
        epoch_pbar = range(num_epochs)
        
    for epoch in epoch_pbar:
        if distributed:
            trainloader.sampler.set_epoch(epoch)
            
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        batch_pbar = tqdm(trainloader, desc=f'Epoch {epoch}', 
                         disable=distributed and rank != 0)
        for inputs, targets in batch_pbar:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if not distributed or rank == 0:
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(valloader)
        val_acc = 100. * val_correct / val_total
        
        # Synchronize metrics across processes if distributed
        if distributed:
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc]).cuda()
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= dist.get_world_size()
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()
        
        # Update learning rate
        scheduler.step()
        
        # Save history and update progress
        if not distributed or rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.2f}%',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log to wandb if available
            try:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            except:
                pass
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch
                
                if experiment_name:
                    dirname = os.path.join(MODEL_SAVE_DIR, experiment_name)
                    os.makedirs(dirname, exist_ok=True)
                    torch.save(best_model_state, 
                             os.path.join(dirname, f'{experiment_name}_best.pth'))
    
    # Load best model for test evaluation
    if best_model_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    if not distributed or rank == 0:
        test_pbar = tqdm(testloader, desc='Testing best model')
    else:
        test_pbar = testloader
        
    with torch.no_grad():
        for inputs, targets in test_pbar:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            if not distributed or rank == 0:
                current_acc = 100. * test_correct / test_total
                test_pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
    
    # Synchronize test metrics if distributed
    if distributed:
        metrics = torch.tensor([test_correct, test_total]).cuda()
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        test_correct, test_total = metrics.tolist()
    
    test_acc = 100. * test_correct / test_total
    
    if not distributed or rank == 0:
        history['test_acc'] = test_acc
        print(f'\nBest Validation Accuracy: {best_val_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        # Log final metrics to wandb
        try:
            wandb.log({
                'best_val_acc': best_val_acc,
                'test_acc': test_acc
            })
        except:
            pass
    
    return history

def cleanup_distributed():
    """
    Cleanup distributed training
    """
    dist.destroy_process_group()

def plot_training_curves(history):
    """
    Plot training curves and return the figure
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['val_losses'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train')
    plt.plot(history['val_accs'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf() 