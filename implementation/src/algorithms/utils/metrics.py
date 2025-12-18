"""
Metrics tracking utilities for neural network training experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class MetricsTracker:
    """
    Tracks training and evaluation metrics throughout training.
    
    This class provides a centralized way to record and access various
    training metrics including losses, accuracies, learning rates, and timing.
    
    Attributes:
        train_losses: Per-epoch average training loss
        train_accs: Per-epoch training accuracy (percentage)
        test_losses: Per-epoch test loss
        test_accs: Per-epoch test accuracy (percentage)
        learning_rates: Learning rate at each epoch
        epoch_times: Time taken for each epoch (seconds)
        batch_losses: Loss per batch (for detailed analysis)
        extra_metrics: Dictionary for any additional metrics
    """
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    test_accs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    batch_losses: List[float] = field(default_factory=list)
    extra_metrics: Dict[str, List[Any]] = field(default_factory=dict)
    
    def record_batch_loss(self, loss: float) -> None:
        """Record loss for a single batch."""
        self.batch_losses.append(loss)
    
    def record_epoch(
        self,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
        lr: float,
        epoch_time: float
    ) -> None:
        """Record all metrics for a completed epoch."""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def record_extra(self, name: str, value: Any) -> None:
        """Record an extra metric by name."""
        if name not in self.extra_metrics:
            self.extra_metrics[name] = []
        self.extra_metrics[name].append(value)
    
    def get_best_test_acc(self) -> tuple[float, int]:
        """
        Get the best test accuracy and the epoch it occurred.
        
        Returns:
            Tuple of (best_accuracy, epoch_index)
        """
        if not self.test_accs:
            return 0.0, -1
        best_idx = int(np.argmax(self.test_accs))
        return self.test_accs[best_idx], best_idx
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dictionary of training results."""
        best_acc, best_epoch = self.get_best_test_acc()
        return {
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_train_acc': self.train_accs[-1] if self.train_accs else None,
            'final_test_loss': self.test_losses[-1] if self.test_losses else None,
            'final_test_acc': self.test_accs[-1] if self.test_accs else None,
            'best_test_acc': best_acc,
            'best_test_epoch': best_epoch + 1,  # 1-indexed
            'total_time_minutes': sum(self.epoch_times) / 60 if self.epoch_times else 0,
            'avg_epoch_time_seconds': np.mean(self.epoch_times) if self.epoch_times else 0,
            'n_epochs': len(self.train_losses),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to a dictionary for serialization."""
        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'batch_losses': self.batch_losses,
            'extra_metrics': self.extra_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsTracker':
        """Create a MetricsTracker from a dictionary."""
        tracker = cls()
        tracker.train_losses = data.get('train_losses', [])
        tracker.train_accs = data.get('train_accs', [])
        tracker.test_losses = data.get('test_losses', [])
        tracker.test_accs = data.get('test_accs', [])
        tracker.learning_rates = data.get('learning_rates', [])
        tracker.epoch_times = data.get('epoch_times', [])
        tracker.batch_losses = data.get('batch_losses', [])
        tracker.extra_metrics = data.get('extra_metrics', {})
        return tracker
