import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from algorithms.utils.metrics import MetricsTracker
from algorithms.nn.optim.sgda import SGDAOptimizer

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset.
    Returns: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, leave=False, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def train_sgd(model, train_loader, test_loader, device, 
              n_epochs=100, lr=0.2, momentum=0.9, weight_decay=5e-4,
              lr_milestones=[30, 75], lr_gamma=0.1, warmup_epochs=0):
    """
    Train model using SGD with momentum and step learning rate decay.
    This is the standard training recipe for ResNet on CIFAR-10.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        device: torch device
        n_epochs: Number of training epochs
        lr: Initial learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
        lr_milestones: Epochs at which to decay learning rate
        lr_gamma: Learning rate decay factor
    
    Returns:
        MetricsTracker with all recorded metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    
    metrics = MetricsTracker()
    
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 80)
    
    for epoch in (range(n_epochs)):
        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{n_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track batch loss
            metrics.record_batch_loss(loss.item())
            
            # Accumulate statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Compute epoch metrics
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - epoch_start
        
        # Store metrics
        metrics.record_epoch(train_loss, train_acc, test_loss, test_acc, current_lr, epoch_time)
        
        print(f"{epoch+1:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {test_loss:>9.4f} | {test_acc:>7.2f}% | {current_lr:>10.6f} | {epoch_time:>5.1f}s")
    
    print("-" * 80)
    print(f"Training complete. Best test accuracy: {max(metrics.test_accs):.2f}%")
    
    return metrics

def train_sgda(model, train_loader, test_loader, device, 
               n_epochs=100, lr=0.2, sigma=0.5, kappa=0.75, momentum=0.9, weight_decay=5e-4, warmup_epochs=0):
    """
    Train model using the custom SGDA optimizer.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        device: torch device
        n_epochs: Number of training epochs
        lr: Initial learning rate (SGDA typically starts higher than SGD, e.g. 0.5 or 1.0)
        sigma: Armijo condition parameter (0 < sigma < 1)
        kappa: Learning rate decay factor (0 < kappa < 1)
        momentum: Momentum factor
        weight_decay: L2 regularization
    
    Returns:
        MetricsTracker with all recorded metrics
    """
    criterion = nn.CrossEntropyLoss()
    
    # Initialize SGDA instead of SGD
    optimizer = SGDAOptimizer(model.parameters(), lr=lr, sigma=sigma, kappa=kappa, momentum=momentum, weight_decay=weight_decay)
    
    metrics = MetricsTracker()
    
    # Added 'Decays' column to track how often the Armijo condition fails
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'LR':>10} | {'Decays':>6} | {'Time':>6}")
    print("-" * 95)
    
    for epoch in range(n_epochs):
        # Warmup
        if epoch < warmup_epochs:
            optimizer.current_lr = lr * (epoch + 1) / warmup_epochs

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        decay_count = 0 # Track how often SGDA reduces LR this epoch
        epoch_start = time.time()
        
        for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{n_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Standard Forward Pass & Gradient Calculation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # CAPTURE LOSS_BEFORE (Scalar)
            loss_before = loss.item()
            
            # 2. Optimization Step
            # SGDA updates weights and returns inner product: lr * <grad, update_dir>
            inner_product = optimizer.step()
            
            # [cite_start]3. Lookahead Forward Pass (No Grad) [cite: 198-202]
            # We must evaluate f(x_{k+1}) to check if the step was valid according to Armijo
            # model.eval()
            with torch.no_grad():
                outputs_new = model(inputs)
                loss_new = criterion(outputs_new, labels)
                loss_after = loss_new.item()
            # model.train()
            # 4. Adaptive Check
            # Check Armijo condition and update LR if needed
            if epoch >= warmup_epochs:
                is_satisfied = optimizer.check_armijo_and_update_lr(loss_before, loss_after, inner_product)
                if not is_satisfied:
                    print("  Armijo condition not satisfied; reducing learning rate.", loss_before, loss_after, inner_product)
                    decay_count += 1
            
            # Track batch loss (using the original loss for consistency)
            metrics.record_batch_loss(loss_before)
            
            # Accumulate statistics
            running_loss += loss_before * inputs.size(0)
            _, predicted = outputs.max(1) # Note: we use original outputs for stats
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Compute epoch metrics
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Get current learning rate (it might have changed multiple times during the epoch)
        current_lr = optimizer.current_lr
        # Handle tensor LR if necessary
        if isinstance(current_lr, torch.Tensor):
            current_lr = current_lr.item()
            
        # Record epoch time
        epoch_time = time.time() - epoch_start
        
        # Store metrics
        metrics.record_epoch(train_loss, train_acc, test_loss, test_acc, current_lr, epoch_time)
        
        print(f"{epoch+1:>5} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {test_loss:>9.4f} | {test_acc:>7.2f}% | {current_lr:>10.6f} | {decay_count:>6} | {epoch_time:>5.1f}s")
    
    print("-" * 95)
    print(f"Training complete. Best test accuracy: {max(metrics.test_accs):.2f}%")
    
    return metrics
