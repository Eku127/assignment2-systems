#!/usr/bin/env python3
"""
Naive implementation of Distributed Data Parallel (DDP) training.

This module provides a simple DDP wrapper that:
1. Broadcasts model parameters from rank 0 to all other ranks at initialization
2. Synchronizes gradients across all ranks using all-reduce after backward pass
3. Averages gradients across all ranks before optimizer step
"""

import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    """
    A naive implementation of Distributed Data Parallel training.
    
    This wrapper handles:
    - Parameter broadcasting from rank 0 to all ranks at initialization
    - Gradient synchronization via all-reduce after backward pass
    
    Usage:
        model = MyModel()
        ddp_model = NaiveDDP(model)
        # ... training loop ...
        loss.backward()
        ddp_model.sync_gradients()  # Call after backward, before optimizer.step()
        optimizer.step()
    """
    
    def __init__(self, module: nn.Module):
        """
        Initialize the DDP wrapper.
        
        Args:
            module: The underlying model to wrap with DDP.
        """
        super().__init__()
        self.module = module
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        """
        Broadcast all parameters from rank 0 to all other ranks.
        This ensures all ranks start with identical model parameters.
        """
        for param in self.module.parameters():
            # Broadcast parameter from rank 0 to all ranks
            dist.broadcast(param.data, src=0)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the underlying module.
        
        Args:
            *args: Positional arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module
            
        Returns:
            Output from the module's forward pass
        """
        return self.module(*args, **kwargs)
    
    def sync_gradients(self):
        """
        Synchronize gradients across all ranks using all-reduce.
        
        This method should be called after backward() but before optimizer.step().
        It averages gradients across all ranks by:
        1. Performing all-reduce on each parameter's gradient
        2. Dividing by world_size to get the average
        
        Only parameters with requires_grad=True are synchronized.
        """
        world_size = dist.get_world_size()
        
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                # All-reduce the gradient across all ranks
                # op=dist.ReduceOp.SUM sums gradients from all ranks
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                
                # Divide by world_size to get the average gradient
                param.grad.data /= world_size
    
    def state_dict(self):
        """Return the state dict of the underlying module."""
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into the underlying module."""
        return self.module.load_state_dict(state_dict)
    
    def named_parameters(self, prefix='', recurse=True):
        """Return named parameters of the underlying module."""
        return self.module.named_parameters(prefix=prefix, recurse=recurse)
    
    def parameters(self, recurse=True):
        """Return parameters of the underlying module."""
        return self.module.parameters(recurse=recurse)


def create_naive_ddp_model(module: nn.Module) -> NaiveDDP:
    """
    Convenience function to create a NaiveDDP wrapper around a module.
    
    Args:
        module: The model to wrap with DDP.
        
    Returns:
        A NaiveDDP instance wrapping the module.
    """
    return NaiveDDP(module)


def sync_gradients_after_backward(model: nn.Module):
    """
    Standalone function to synchronize gradients after backward pass.
    
    This function can be used if you have a NaiveDDP model and want to
    synchronize gradients. It's equivalent to calling model.sync_gradients().
    
    Args:
        model: A NaiveDDP model instance.
    """
    if isinstance(model, NaiveDDP):
        model.sync_gradients()
    else:
        # If it's not a NaiveDDP model, try to sync gradients manually
        world_size = dist.get_world_size()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

