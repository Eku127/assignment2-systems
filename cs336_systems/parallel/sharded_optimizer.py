#!/usr/bin/env python3
"""
Sharded Optimizer implementation for reducing memory consumption in distributed training.

This module provides an optimizer wrapper that:
1. Shards optimizer states across multiple ranks/devices
2. Each rank only updates a subset of parameters (~1/world_size)
3. Synchronizes updated parameters after each optimizer step via broadcast
"""

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any, Iterable, Dict, List


class ShardedOptimizer(Optimizer):
    """
    An optimizer wrapper that shards optimizer states across multiple ranks.
    
    Each rank only stores and updates optimizer states for a subset of parameters,
    significantly reducing memory consumption. After each optimizer step, parameters
    are synchronized across ranks via broadcast operations.
    
    Usage:
        model = MyModel()
        # Wrap with DDP for gradient synchronization
        ddp_model = DDP(model)
        
        # Create sharded optimizer
        optimizer = ShardedOptimizer(
            ddp_model.parameters(),
            optimizer_cls=torch.optim.AdamW,
            lr=1e-4
        )
        
        # Training loop
        loss.backward()
        optimizer.step()  # Automatically synchronizes parameters after update
    """
    
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        Initialize the sharded optimizer.
        
        Args:
            params: An iterable of parameters to optimize or parameter groups
            optimizer_cls: The optimizer class to wrap (e.g., torch.optim.AdamW)
            **kwargs: Additional arguments to pass to the optimizer constructor
        """
        # Get world size and rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Store the optimizer class and kwargs for later use
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        # Convert params to list if it's an iterable
        if isinstance(params, torch.Tensor):
            params = [params]
        elif not isinstance(params, list):
            params = list(params)
        
        # Store all parameters (before sharding) - collect them first
        self._all_params: List[torch.nn.Parameter] = []
        
        # Mapping from parameter to its owner rank
        self._param_to_rank: Dict[torch.nn.Parameter, int] = {}
        
        # Collect all parameters first and assign them to ranks
        all_params_list = []
        if len(params) > 0 and isinstance(params[0], dict):
            # params is a list of parameter groups
            for param_group in params:
                group_params = param_group['params']
                if isinstance(group_params, torch.Tensor):
                    group_params = [group_params]
                else:
                    group_params = list(group_params)
                all_params_list.extend(group_params)
        else:
            # params is a list of parameters
            all_params_list = params
        
        if len(all_params_list) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        # Pre-assign all parameters to ranks BEFORE calling super().__init__
        # This ensures that when add_param_group is called by the base class,
        # _all_params and _param_to_rank are already populated
        for i, param in enumerate(all_params_list):
            self._all_params.append(param)
            assigned_rank = i % self.world_size
            self._param_to_rank[param] = assigned_rank
        
        # Initialize base class with all params (it will call add_param_group)
        # But our add_param_group is already prepared with _all_params and _param_to_rank
        defaults = kwargs.copy()
        super().__init__(all_params_list, defaults)
        
        # Base class __init__ already called add_param_group for all params
        # So we don't need to call it again here
        # But we need to make sure _optimizer gets the right parameters
        
        # Create the underlying optimizer with only the parameters owned by this rank
        owned_params = [p for p in self._all_params if self._param_to_rank[p] == self.rank]
        self._optimizer = optimizer_cls(owned_params, **kwargs)
    
    def add_param_group(self, param_group: dict[str, Any]):
        """
        Add a parameter group to the sharded optimizer.
        
        This method is called by the Optimizer super-class constructor and may also
        be called during training. It handles assigning parameters among ranks.
        
        Args:
            param_group: A dict containing 'params' (iterable of parameters) and
                        optionally other optimizer-specific settings
        """
        # Extract parameters from the param_group
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = list(params)
        
        # Parameters should already be in _all_params and _param_to_rank
        # (assigned during __init__), so we just need to verify and add to base class
        for param in params:
            if param not in self._param_to_rank:
                # This is a new parameter added after initialization
                # Assign it to a rank using round-robin
                param_idx = len(self._all_params)
                self._all_params.append(param)
                assigned_rank = param_idx % self.world_size
                self._param_to_rank[param] = assigned_rank
        
        # Add to super-class param_groups (required by Optimizer interface)
        # Only add if not already added (to avoid duplicates when called from __init__)
        # Check if this param_group is already in base class param_groups
        base_param_groups = object.__getattribute__(self, 'param_groups')
        param_ids_in_base = set()
        for pg in base_param_groups:
            for p in pg['params']:
                param_ids_in_base.add(id(p))
        
        # Only add if any param is not already in base param_groups
        if any(id(p) not in param_ids_in_base for p in params):
            super().add_param_group(param_group)
        
        # If _optimizer has been created, add this rank's parameters to it
        if hasattr(self, '_optimizer'):
            # Extract only the parameters owned by this rank
            owned_params = [p for p in params if self._param_to_rank[p] == self.rank]
            if len(owned_params) > 0:
                # Create a new param_group dict with only owned params
                owned_param_group = param_group.copy()
                owned_param_group['params'] = owned_params
                self._optimizer.add_param_group(owned_param_group)
    
    def step(self, closure=None, **kwargs):
        """
        Perform an optimizer step and synchronize parameters across ranks.
        
        Args:
            closure: Optional closure to reevaluate the model (for some optimizers)
            **kwargs: Additional arguments to pass to the underlying optimizer's step()
        
        Returns:
            The return value from the underlying optimizer's step() (if any)
        """
        # Step 1: Each rank updates its own parameters using the underlying optimizer
        loss = self._optimizer.step(closure, **kwargs)
        
        # Step 2: Synchronize parameters across ranks
        # Each rank broadcasts its updated parameters to all other ranks
        self._synchronize_parameters()
        
        return loss
    
    def _synchronize_parameters(self):
        """
        Synchronize parameters across all ranks after optimizer step.
        
        Each rank broadcasts its updated parameters to all other ranks.
        This ensures all ranks have the same model parameters after the step.
        """
        for param in self._all_params:
            # Get the owner rank for this parameter
            owner_rank = self._param_to_rank[param]
            
            # Broadcast this parameter from its owner to all other ranks
            dist.broadcast(param.data, src=owner_rank)
    
    def __getattribute__(self, name):
        """Override to return _optimizer.param_groups when accessing param_groups."""
        if name == 'param_groups':
            # Return the underlying optimizer's param_groups
            # Use object.__getattribute__ to avoid infinite recursion
            try:
                _optimizer = object.__getattribute__(self, '_optimizer')
                return _optimizer.param_groups
            except AttributeError:
                # Fallback to base class param_groups if _optimizer not yet created
                return super().__getattribute__('param_groups')
        return super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        """Override to allow base class to set param_groups during initialization."""
        if name == 'param_groups' and hasattr(self, '_optimizer'):
            # If _optimizer exists, we want to redirect to it
            # But during __init__, we need to allow base class to set it
            # So we check if we're in __init__ by checking if _optimizer exists
            # Actually, we should just allow it to be set normally
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
    
    def state_dict(self):
        """
        Return the state dict of the underlying optimizer.
        
        Note: This only contains the state for parameters owned by this rank.
        """
        return self._optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """
        Load state dict into the underlying optimizer.
        
        Args:
            state_dict: State dict to load (should only contain state for this rank's parameters)
        """
        return self._optimizer.load_state_dict(state_dict)
    
    def zero_grad(self, set_to_none: bool = True):
        """
        Zero out gradients for all parameters.
        
        Args:
            set_to_none: Whether to set gradients to None instead of zeroing them
        """
        # Zero gradients for all parameters (not just owned ones)
        # This is necessary because gradients are computed for all parameters during backward
        for param in self._all_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()


def create_sharded_optimizer(params, optimizer_cls: Type[Optimizer], **kwargs) -> ShardedOptimizer:
    """
    Convenience function to create a ShardedOptimizer instance.
    
    Args:
        params: An iterable of parameters to optimize or parameter groups
        optimizer_cls: The optimizer class to wrap (e.g., torch.optim.AdamW)
        **kwargs: Additional arguments to pass to the optimizer constructor
        
    Returns:
        A ShardedOptimizer instance
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)

