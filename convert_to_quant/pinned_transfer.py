"""
Pinned memory utilities for faster CPU→GPU tensor transfers.

Pinned (page-locked) memory enables faster DMA transfers to GPU.
Uses PyTorch's native pin_memory() with non_blocking transfers.
"""
import torch
from typing import Optional


def transfer_to_gpu_pinned(
    tensor: torch.Tensor,
    device: str = 'cuda',
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Transfer tensor to GPU using pinned memory for faster transfer.
    
    Pinned memory enables non-blocking DMA transfers which can be 2-3x faster
    than regular pageable memory transfers for large tensors.
    
    Args:
        tensor: CPU tensor to transfer
        device: Target GPU device (default 'cuda')
        dtype: Optional dtype conversion during transfer
        
    Returns:
        Tensor on GPU device
        
    Note:
        Falls back to regular .to() if:
        - Tensor is already on GPU
        - CUDA is not available
        - Pinning fails (e.g., insufficient memory)
    """
    # Skip if not a CPU tensor or CUDA unavailable
    if tensor.device.type != 'cpu' or not torch.cuda.is_available():
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
    
    # Skip if target is not CUDA
    if not str(device).startswith('cuda'):
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
    
    try:
        # Pin memory for faster DMA transfer
        pinned = tensor.pin_memory()
        
        # Non-blocking transfer (overlaps with computation if stream allows)
        if dtype is not None:
            result = pinned.to(device=device, dtype=dtype, non_blocking=True)
        else:
            result = pinned.to(device=device, non_blocking=True)
        
        # Synchronize to ensure transfer is complete before returning
        torch.cuda.current_stream().synchronize()
        
        return result
        
    except Exception:
        # Fall back to regular transfer if pinning fails
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
