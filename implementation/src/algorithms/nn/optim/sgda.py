"""
Stochastic Gradient Descent with Adaptive step-size (SGDA) optimizer for PyTorch.

This implements the Self-adaptive Gradient Descent Algorithm from the paper
"Self-adaptive algorithms for quasiconvex programming and applications to machine learning"
by Tran Ngoc Thang.

The algorithm adapts the step size based on an Armijo-like condition:
    f(x_{k+1}) <= f(x_k) - sigma * <grad_f(x_k), x_k - x_{k+1}>

If the condition is violated, the step size is reduced by a factor of kappa.
"""

import torch
from torch.optim import Optimizer
from typing import Union, Optional, Callable
from torch import Tensor


class SGDAOptimizer(Optimizer):
    def __init__(
        self,
        params,
        sigma: Union[float, Tensor] = 0.1,
        kappa: Union[float, Tensor] = 0.75,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0.0,
        weight_decay: Union[float, Tensor] = 0,
    ):
        self.sigma = sigma
        self.kappa = kappa
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Perform a single optimization step and returns the inner product of the Armijo Condition"""
        inner_product = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Momentum
                param_state = self.state[p]
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)

                    # Update direction is the momentum buffer
                    update_direction = buf
                else:
                    update_direction = d_p

                # Calculate inner product <grad, step_direction>
                # step_direction = x_k - x_{k+1} = lr * update_direction
                # inner_product += <d_p, lr * update_direction>

                # We use the original gradient d_p for the inner product calculation as per paper/standard Armijo
                # <nabla f(x_k), x_k - x_{k+1}>
                term = torch.dot(d_p.view(-1), update_direction.view(-1))
                inner_product += lr * term.item()

                p.data.add_(update_direction, alpha=-lr)

        return inner_product

    @property
    def current_lr(self):
        """Get the current learning rate."""
        return self.param_groups[0]["lr"]

    @current_lr.setter
    def current_lr(self, new_lr):
        """Set a new learning rate for all parameter groups."""
        for group in self.param_groups:
            group["lr"] = new_lr

    def check_armijo_and_update_lr(self, loss_before, loss_after, inner_product):
        """Check the Armijo condition and update the learning rate accordingly.

        Args:
            loss_before: Loss before the optimization step
            loss_after: Loss after the optimization step
            inner_product: Inner product computed during the step

        Returns:
            bool: True if Armijo condition is satisfied, False otherwise

        If the Armijo condition is not satisfied, the learning rate is reduced by multiplying with kappa.
        """

        armijo = loss_after - loss_before + self.sigma * inner_product
        if armijo > 0:
            # Condition failed: reduce learning rate
            self.current_lr *= self.kappa
            return False

        return True


__all__ = ["SGDAOptimizer"]
