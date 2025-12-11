import argparse
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import gc
import math
import json
from torch.optim import AdamW, RAdam
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
from quant_ops import BlockWiseINT8Layout

# Triton V2 kernels (alternative INT8 quantization backend)
try:
    from kernels.int8_matmul import (
        weight_quant as triton_v2_weight_quant,
        weight_dequant as triton_v2_weight_dequant,
    )
    _HAS_TRITON_V2 = True
except ImportError:
    _HAS_TRITON_V2 = False

# --- Constants and Configuration ---
torch.set_printoptions(precision=8)
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared", "patch_embedding", "audio_model.patch_embedding", "ref_conv", "control_adapter", "motion_encoder.enc.net_app", "face_encoder.conv", "pose_patch_embedding", "motion_encoder.enc.fc", "img_emb.proj", "q_norm", "motion_encoder.dec", "head.modulation", "casual_audio_encoder", "cond_encoder", "frame_packer", "norm_k", "norm_q"]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
HUNYUAN_AVOID_KEY_NAMES = ["layernorm", "img_attn_k_norm", "img_attn_q_norm", "txt_attn_k_norm", "txt_attn_q_norm", "norm1", "norm2", "vision_in.proj.0", "vision_in.proj.4", "img_in.proj", "cond_type_embedding"]
ZIMAGE_AVOID_KEY_NAMES = ["cap_embedder.0", "cap_pad_token", "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2", "k_norm", "q_norm", "x_pad_token"]
DISTILL_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder", "txt_in"]
NERF_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder"]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv", "__x0__"]
WAN_LAYER_KEYNAMES = ["text_embedding", "time_embedding", "audio_model.text_embedding", "audio_model.time_embedding", "time_projection", "video_model.time_projection", "head.head", "face_encoder.out_proj", "face_adapter"]
QWEN_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]
ZIMAGE_LAYER_KEYNAMES = ["x_embedder", "final_layer", "cap_embedder.1", "adaLN_modulation", "t_embedder"]
TARGET_FP8_DTYPE = torch.float8_e4m3fn
TARGET_INT8_DTYPE = torch.int8
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# FP8 constants
FP8_MIN = float(torch.finfo(TARGET_FP8_DTYPE).min)
FP8_MAX = float(torch.finfo(TARGET_FP8_DTYPE).max)
FP8_MIN_POS = float(torch.finfo(TARGET_FP8_DTYPE).tiny)

# INT8 constants (using symmetric range [-127, 127] for symmetric quantization)
INT8_MIN = int(torch.iinfo(TARGET_INT8_DTYPE).min)  # -128
INT8_MAX = int(torch.iinfo(TARGET_INT8_DTYPE).max)  # 127
INT8_SYMMETRIC_MAX = min(abs(INT8_MIN), INT8_MAX)   # 127 (symmetric range)

def dict_to_tensor(data_dict):
    json_str = json.dumps(data_dict)
    byte_data = json_str.encode('utf-8')
    tensor_data = torch.tensor(list(byte_data), dtype=torch.uint8)
    return tensor_data

def tensor_to_dict(tensor_data):
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode('utf-8')
    data_dict = json.loads(json_str)
    return data_dict


def create_comfy_quant_tensor(format_type: str, block_size: Optional[int] = None, full_precision_matrix_mult: Optional[bool] = None) -> torch.Tensor:
    """
    Create a .comfy_quant tensor for ComfyUI quantization metadata.
    
    Args:
        format_type: Either "float8_e4m3fn" or "int8_blockwise"
        block_size: Block/group size for quantization (required for int8_blockwise)
        full_precision_matrix_mult: If True, adds "full_precision_matrix_mult": True to metadata.
                                    If False or None, this key is omitted entirely.
    
    Returns:
        torch.uint8 tensor containing JSON-encoded metadata
    """
    comfy_quant = {"format": format_type}
    
    if format_type == "int8_blockwise" and block_size is not None:
        comfy_quant["group_size"] = block_size
    
    if full_precision_matrix_mult is True:
        comfy_quant["full_precision_matrix_mult"] = True
    
    return dict_to_tensor(comfy_quant)


def should_skip_layer_for_performance(tensor: torch.Tensor, block_size: int) -> Tuple[bool, str]:
    """
    Check if a layer should be skipped based on performance heuristics.
    
    Args:
        tensor: Weight tensor to evaluate
        block_size: Block size for quantization
    
    Returns:
        Tuple of (should_skip, reason)
    """
    if tensor.ndim != 2:
        return True, "not 2D"
    
    rows, cols = tensor.shape
    
    # Skip if any dimension is smaller than block_size
    if rows < block_size or cols < block_size:
        return True, f"dimension smaller than block_size ({block_size})"
    
    # Skip if dimensions are not divisible by block_size
    if rows % block_size != 0 or cols % block_size != 0:
        return True, f"dimensions not divisible by block_size ({block_size})"
    
    # Skip highly rectangular layers (aspect ratio > 4.0 with small dimension < 5120)
    aspect_ratio = max(rows, cols) / max(min(rows, cols), 1)
    small_dim = min(rows, cols)
    if aspect_ratio > 4.0 and small_dim < 5120:
        return True, f"highly rectangular (aspect ratio {aspect_ratio:.1f}, small dim {small_dim})"
    
    # Skip small-ish layers (< 15M params, aspect ratio < 2.0)
    num_params = rows * cols
    if num_params < 15_000_000 and aspect_ratio < 2.0:
        return True, f"small layer ({num_params:,} params, aspect ratio {aspect_ratio:.1f})"
    
    # Skip layers with max dimension < 4096
    if max(rows, cols) < 4096:
        return True, f"max dimension < 4096 ({max(rows, cols)})"
    
    return False, ""


class LearnedRoundingConverter:
    """
    Implements advanced quantization using learned adaptive rounding.
    Provides a highly effective optimization strategy.
    Supports both FP8 and INT8 quantization formats.
    """
    def __init__(self, optimizer="original", num_iter=500, top_p=0.01, min_k=1, max_k=16, scaling_mode='tensor', block_size=64, full_matrix=False, target_format='fp8', no_learned_rounding=False, kernel_backend='triton', **kwargs):
        self.num_iter = num_iter
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.block_size = block_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer_choice = optimizer
        self.full_matrix = full_matrix
        self.target_format = target_format
        self.no_learned_rounding = no_learned_rounding
        self.kernel_backend = kernel_backend
        self.optimizer_kwargs = kwargs
        
        # INT8 always uses block-wise scaling
        if target_format == 'int8':
            scaling_mode = 'block'
        self.scaling_mode = scaling_mode
        
        # Set format-specific max values
        if self.target_format == 'int8':
            # INT8 uses integer symmetric range [-127, 127]
            # Note: quant_max_val is not used in INT8 path - BlockWiseINT8Layout handles quantization
            self.target_dtype = TARGET_INT8_DTYPE
            # INT8 uses BlockWiseINT8Layout which computes: scale = amax / 127.0
            # The 127 is used directly as integer, not converted to float for max comparison
            self.f8_max_val = None  # Not applicable to INT8
        else:
            # FP8 uses floating point range constants
            self.target_dtype = TARGET_FP8_DTYPE
            self.f8_max_val = FP8_MAX  # Used in FP8 scale calculation and clamping

        print(f"LearnedRoundingConverter initialized on device: {self.device}")
        print(f"  - Target format: {self.target_format}")
        print(f"  - Using optimizer: '{self.optimizer_choice}'" + (" (disabled - simple quant)" if self.no_learned_rounding else ""))
        print(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode == 'block':
            print(f"    - Block size: {self.block_size}")
        if self.target_format == 'int8':
            print(f"  - Kernel backend: {self.kernel_backend}")

    def _optimize_ppsf(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = ProdigyPlusScheduleFree([delta], lr=lr, betas=(0.9, 0.99), beta3=None, 
                                            weight_decay=0.0, weight_decay_by_lr=False, d0=1e-3, d_coef=1.0,
                                            d_limiter=True, prodigy_steps=0, schedulefree_c=0, eps=1e-8,
                                            split_groups=False, split_groups_mean=False,
                                            factored=True, factored_fp32=True, use_bias_correction=False,
                                            use_stableadamw=True, use_schedulefree=True, use_speed=False,
                                            stochastic_rounding=True, fused_back_pass=False,
                                            use_cautious=False, use_grams=False, use_adopt=False,
                                            use_orthograd=False, use_focus=False)
        best_loss = float('inf')
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (ProdigyPlusScheduleFree)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()

            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
            if best_loss < 1e-8:
                print(f"      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_adamw(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = AdamW([delta], lr=lr)
        best_loss = float('inf')
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (AdamW)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()

            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
            if best_loss < 1e-8:
                print(f"      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_radam(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = RAdam([delta], lr=lr)
        best_loss = float('inf')
        best_delta = delta.detach().clone()

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (RAdam)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()

            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
            if best_loss < 1e-8:
                print(f"      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_original(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        W_q_refined = W_rounded.clone()
        best_loss = float('inf')
        best_tensor = None
        worse_loss_counter = 0
        curr_lr = self.optimizer_kwargs.get('lr', 0.5)
        if W_float32.shape[0] == W_float32.shape[1]:
            small_mult = 0.95
        else:
            small_mult = 1.0

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (Original)", leave=False, dynamic_ncols=True)
        for i in pbar:
            with torch.no_grad():
                current_dq = W_q_refined / scale
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            if loss.item() < best_loss and worse_loss_counter < 50:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.25 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 49 and worse_loss_counter < 75:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.375 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 74 and worse_loss_counter < 100:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.5 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 99 and worse_loss_counter < 125:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.75 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 124 and worse_loss_counter < 150:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (2.0 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 149 and worse_loss_counter < 200:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (2.25 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 199 and worse_loss_counter < 250:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (2.5 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 249 and worse_loss_counter < 300:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (2.75 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 299:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (3.0 * small_mult), 100.0)
            elif loss.item() > best_loss and worse_loss_counter < 26:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.95 * small_mult), 9e-8)
            elif worse_loss_counter > 25 and worse_loss_counter < 51:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.97 * small_mult), 8e-8)
            elif worse_loss_counter > 50 and worse_loss_counter < 76:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.985 * small_mult), 7e-8)
            elif worse_loss_counter > 75 and worse_loss_counter < 101:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.9875 * small_mult), 6e-8)
            elif worse_loss_counter > 100 and worse_loss_counter < 151:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.98875 * small_mult), 5e-8)
            elif worse_loss_counter > 150 and worse_loss_counter < 201:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.99 * small_mult), 4e-8)
            elif worse_loss_counter > 200 and worse_loss_counter < 251:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.99125 * small_mult), 3e-8)
            elif worse_loss_counter > 250 and worse_loss_counter < 301:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.9925 * small_mult), 2e-8)
            elif worse_loss_counter > 300:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.995 * small_mult), 5e-9)
        
        
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}", "worse_count": f"{worse_loss_counter}"})
        
            if loss.item() < 1e-8 or curr_lr < 1e-08 or worse_loss_counter > 500:
                if curr_lr < 1.75e-08 and worse_loss_counter > 450:
                    print("      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif loss.item() < 1e-8 and curr_lr < 1.75e-8:
                    print("      - Learning Rate has bottomed out and loss is negligible. Stopping.")
                elif worse_loss_counter > 450 and loss.item() > 2e-8:
                    print("      - Loss is negligible and loss has stalled. Stopping.")
                elif loss.item() < 1e-8:
                    print("      - Loss is negligible. Stopping.")
                elif curr_lr < 1e-08:
                    print("      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > 500:
                    print("      - Loss has stalled. Stopping.")
                break
        
            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                W_q_refined -= curr_lr * (grad_direction * scale)
        
        pbar.close()
        return best_tensor if best_tensor is not None else W_q_refined

    def convert(self, W_orig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            print("  - Tensor is all zeros, skipping optimization.")
            quantized_tensor = torch.zeros_like(W_float32, dtype=self.target_dtype)
            dequant_scale = None
            if self.scaling_mode == 'block' and W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                out_features, in_features = W_float32.shape
                if self.target_format == 'int8':
                    # INT8 uses 2D block scaling (M//block_size, N//block_size)
                    num_blocks_m = out_features // self.block_size
                    num_blocks_n = in_features // self.block_size
                    dequant_scale = torch.ones(num_blocks_m, num_blocks_n, device=self.device, dtype=SCALE_DTYPE)
                else:
                    num_blocks = in_features // self.block_size
                    dequant_scale = torch.ones(out_features, num_blocks, 1, device=self.device, dtype=SCALE_DTYPE)
            else:
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            return quantized_tensor, dequant_scale, torch.zeros_like(W_float32)

        # INT8 quantization path
        if self.target_format == 'int8':
            return self._convert_int8(W_float32)
        
        # FP8 quantization path (original behavior)
        return self._convert_fp8(W_float32)

    def _convert_int8(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        INT8 block-wise quantization using BlockWiseINT8Layout or triton_v2 kernels.
        
        INT8 block-wise quantization differs from FP8:
        - Uses symmetric quantization with range [-127, 127]
        - Scale is per-block (2D grid): shape (M//block_size, N//block_size)
        - Requires dimensions divisible by block_size
        """
        M, N = W_float32.shape
        
        # Validate dimensions are divisible by block_size
        if M % self.block_size != 0 or N % self.block_size != 0:
            raise ValueError(
                f"INT8 block-wise quantization requires dimensions divisible by block_size={self.block_size}. "
                f"Got shape ({M}, {N}). Consider using --skip_inefficient_layers or a different block_size."
            )
        
        # Select quantization backend
        if self.kernel_backend == 'triton_v2':
            # Use triton_v2 kernels from int8_matmul.py
            qdata, scale = triton_v2_weight_quant(W_float32, block_size=self.block_size)
        else:
            # Use BlockWiseINT8Layout (default 'triton' backend from quant_ops.py)
            qdata, layout_params = BlockWiseINT8Layout.quantize(
                W_float32,
                block_size=self.block_size,
                is_weight=True
            )
            scale = layout_params['scale']  # Shape: (M//block_size, N//block_size)

        # Optional: Apply learned rounding optimization for INT8
        if not self.no_learned_rounding and self.num_iter > 0:
            print(f"    - Applying learned rounding optimization for INT8...")
            qdata, scale = self._optimize_int8_learned_rounding(W_float32, qdata, scale)
        
        # Dequantize to get the reconstructed weight for bias correction
        if self.kernel_backend == 'triton_v2':
            dequantized_weight = triton_v2_weight_dequant(qdata, scale, block_size=self.block_size, output_dtype=COMPUTE_DTYPE)
        else:
            dequantized_weight = BlockWiseINT8Layout.dequantize(
                qdata, scale, self.block_size, is_weight=True, orig_dtype=COMPUTE_DTYPE
            )
        
        # Clean up
        del W_float32
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight

    def _int8_dequantize_blockwise(self, qdata: torch.Tensor, scale: torch.Tensor, M: int, N: int, block_size: int) -> torch.Tensor:
        """
        Differentiable block-wise INT8 dequantization for optimization.
        Matches BlockWiseINT8Layout._weight_quantize_pytorch logic.
        
        Args:
            qdata: Quantized values (can be float during optimization), shape (M, N)
            scale: Per-block scales, shape (M//block_size, N//block_size)
            M, N: Original tensor dimensions
            block_size: Block size for quantization
        
        Returns:
            Dequantized tensor, shape (M, N)
        """
        # Reshape to blocks: (M//bs, bs, N//bs, bs)
        q_blocked = qdata.reshape(M // block_size, block_size, N // block_size, block_size)
        # Permute to: (M//bs, N//bs, bs, bs)
        q_blocked = q_blocked.permute(0, 2, 1, 3)
        # Broadcast scale: (M//bs, N//bs, 1, 1)
        scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
        # Apply scale
        dequantized = q_blocked * scale_broadcast
        # Permute back and reshape: (M, N)
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    def _optimize_int8_learned_rounding(self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned rounding optimization for INT8 quantization.
        Uses SVD-based optimization similar to FP8 but adapted for INT8.
        Supports multiple optimizer choices: original, adamw, radam, ppsf.
        """
        M, N = W_float32.shape
        block_size = self.block_size
        
        # Compute SVD for the optimization
        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)
        
        print(f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components.")
        
        if self.full_matrix:
            print(f"    - Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver='gesvd')
        else:
            try:
                print(f"    - Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)
        
        U_k, Vh_k = U[:, :k], Vh[:k, :]
        
        # Route to appropriate optimizer
        if self.optimizer_choice == 'original':
            final_qdata = self._optimize_int8_original(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == 'adamw':
            final_qdata = self._optimize_int8_adamw(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == 'radam':
            final_qdata = self._optimize_int8_radam(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == 'ppsf':
            final_qdata = self._optimize_int8_ppsf(W_float32, qdata, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")
        
        del U, Vh, U_k, Vh_k
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return final_qdata, scale

    def _optimize_int8_adamw(self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        """INT8 optimization using AdamW optimizer."""
        M, N = W_float32.shape
        block_size = self.block_size
        
        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)
        
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = AdamW([delta], lr=lr)
        best_loss = float('inf')
        best_delta = delta.detach().clone()
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing INT8 (AdamW)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            
            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
            
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)
            
            loss.backward()
            optimizer.step()
            
            current_loss_val = loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
            
            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
            if best_loss < 1e-8:
                print(f"      - Loss is negligible. Stopping early.")
                break
        
        pbar.close()
        
        final_qdata = (qdata_float + best_delta).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).round().to(TARGET_INT8_DTYPE)
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_radam(self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        """INT8 optimization using RAdam optimizer."""
        M, N = W_float32.shape
        block_size = self.block_size
        
        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)
        
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = RAdam([delta], lr=lr)
        best_loss = float('inf')
        best_delta = delta.detach().clone()
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing INT8 (RAdam)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            
            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
            
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)
            
            loss.backward()
            optimizer.step()
            
            current_loss_val = loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
            
            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
            if best_loss < 1e-8:
                print(f"      - Loss is negligible. Stopping early.")
                break
        
        pbar.close()
        
        final_qdata = (qdata_float + best_delta).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).round().to(TARGET_INT8_DTYPE)
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_ppsf(self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        """INT8 optimization using ProdigyPlusScheduleFree optimizer."""
        M, N = W_float32.shape
        block_size = self.block_size
        
        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)
        
        lr = self.optimizer_kwargs.get('lr', 1e-2)
        optimizer = ProdigyPlusScheduleFree([delta], lr=lr, betas=(0.9, 0.99), beta3=None, 
                                            weight_decay=0.0, weight_decay_by_lr=False, d0=1e-3, d_coef=1.0,
                                            d_limiter=True, prodigy_steps=0, schedulefree_c=0, eps=1e-8,
                                            split_groups=False, split_groups_mean=False,
                                            factored=True, factored_fp32=True, use_bias_correction=False,
                                            use_stableadamw=True, use_schedulefree=True, use_speed=False,
                                            stochastic_rounding=True, fused_back_pass=False,
                                            use_cautious=False, use_grams=False, use_adopt=False,
                                            use_orthograd=False, use_focus=False)
        best_loss = float('inf')
        best_delta = delta.detach().clone()
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing INT8 (ProdigyPlusScheduleFree)", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            
            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
            
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)
            
            loss.backward()
            optimizer.step()
            
            current_loss_val = loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
            
            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}"})
            if best_loss < 1e-8:
                print(f"      - Loss is negligible. Stopping early.")
                break
        
        pbar.close()
        
        final_qdata = (qdata_float + best_delta).clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).round().to(TARGET_INT8_DTYPE)
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_original(self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        """INT8 optimization using original gradient-based optimizer (no autograd)."""
        M, N = W_float32.shape
        block_size = self.block_size
        
        qdata_float = qdata.to(COMPUTE_DTYPE)
        q_refined = qdata_float.clone()
        
        best_loss = float('inf')
        best_tensor = None
        worse_loss_counter = 0
        curr_lr = self.optimizer_kwargs.get('lr', 0.5)
        
        if M == N:
            small_mult = 0.95
        else:
            small_mult = 1.0
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing INT8 (Original)", leave=False, dynamic_ncols=True)
        for i in pbar:
            with torch.no_grad():
                current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)
            
            # Adaptive learning rate logic (same as FP8 original optimizer)
            if loss.item() < best_loss and worse_loss_counter < 50:
                best_loss = loss.item()
                best_tensor = q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.25 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 49 and worse_loss_counter < 75:
                best_loss = loss.item()
                best_tensor = q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.375 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 74 and worse_loss_counter < 100:
                best_loss = loss.item()
                best_tensor = q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.5 * small_mult), 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 99:
                best_loss = loss.item()
                best_tensor = q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * (1.75 * small_mult), 100.0)
            elif loss.item() > best_loss and worse_loss_counter < 26:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.95 * small_mult), 9e-8)
            elif worse_loss_counter > 25 and worse_loss_counter < 51:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.97 * small_mult), 8e-8)
            elif worse_loss_counter > 50 and worse_loss_counter < 76:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.985 * small_mult), 7e-8)
            elif worse_loss_counter > 75 and worse_loss_counter < 101:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.9875 * small_mult), 6e-8)
            elif worse_loss_counter > 100:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * (0.99 * small_mult), 5e-8)
            
            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}", "worse_count": f"{worse_loss_counter}"})
            
            if loss.item() < 1e-8 or curr_lr < 1e-08 or worse_loss_counter > 500:
                if curr_lr < 1.75e-08 and worse_loss_counter > 450:
                    print("      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif loss.item() < 1e-8:
                    print("      - Loss is negligible. Stopping.")
                elif curr_lr < 1e-08:
                    print("      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > 500:
                    print("      - Loss has stalled. Stopping.")
                break
            
            with torch.no_grad():
                # Compute gradient direction in INT8 quantized space
                # 
                # Math derivation:
                # - Dequantization: dq = Q * scale (per-block)
                # - Loss L is computed on dq
                # - By chain rule: ∂L/∂Q = ∂L/∂dq * ∂dq/∂Q = ∂L/∂dq * scale
                # 
                # So we need to MULTIPLY the weight-space gradient by scale to get Q-space gradient
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                
                # Transform gradient through block-wise structure
                # Reshape grad to blocks, multiply by scale (chain rule), then reshape back
                grad_blocked = grad_direction.reshape(M // block_size, block_size, N // block_size, block_size)
                grad_blocked = grad_blocked.permute(0, 2, 1, 3)
                scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
                grad_scaled = grad_blocked * scale_broadcast
                grad_scaled = grad_scaled.permute(0, 2, 1, 3).reshape(M, N)
                
                q_refined -= curr_lr * grad_scaled
        
        pbar.close()
        
        final_tensor = best_tensor if best_tensor is not None else q_refined
        final_qdata = final_tensor.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX).round().to(TARGET_INT8_DTYPE)
        del qdata_float, q_refined
        return final_qdata

    def _convert_fp8(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original FP8 quantization path."""

        scale = None
        compact_scale = None
        current_scaling_mode = self.scaling_mode

        if current_scaling_mode == 'block':
            if W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                print(f"    - Using block scaling with block size {self.block_size}.")
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                W_reshaped = W_float32.view(out_features, num_blocks, self.block_size)
                w_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]
                compact_scale = self.f8_max_val / w_max.clamp_min_(1e-12)
                scale = compact_scale.repeat_interleave(self.block_size, dim=2).view(out_features, in_features)
            else:
                print(f"    - WARNING: Tensor shape {list(W_float32.shape)} not suitable for block size {self.block_size}. Falling back to 'tensor' scaling.")
                current_scaling_mode = 'tensor'

        if current_scaling_mode == 'tensor':
            w_max = W_float32.abs().max()
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            compact_scale = scale
        
        assert scale is not None, "scale should not be None after scaling mode selection"

        # Skip SVD optimization if no_learned_rounding is set
        if self.no_learned_rounding:
            print(f"    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                W_f8 = (W_float32 * scale).clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
                if compact_scale is None:
                    dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
                else:
                    if current_scaling_mode == 'block':
                        dequant_scale = compact_scale.reciprocal()
                    else:
                        dequant_scale = compact_scale.reciprocal().reshape(1)
                    dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
                dequantized_weight_tensor = (W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale)
            del W_float32, scale, compact_scale
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return W_f8, dequant_scale, dequantized_weight_tensor

        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components.")

        if self.full_matrix == True:
            print(f"Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver='gesvd')
        else:
            try:
                print(f"Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        if self.optimizer_choice == 'ppsf':
            final_tensor_scaled = self._optimize_ppsf(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        elif self.optimizer_choice == 'adamw':
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        elif self.optimizer_choice == 'radam':
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        elif self.optimizer_choice == 'original':
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
            final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        #    final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        #    final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)

        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(TARGET_FP8_DTYPE)
            # Ensure compact_scale is valid before calling reciprocal; fall back to ones if missing.
            if compact_scale is None:
                print("    - WARNING: compact_scale is None, falling back to torch.ones for dequant_scale.")
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                if current_scaling_mode == 'block':
                    dequant_scale = compact_scale.reciprocal()
                else:
                    dequant_scale = compact_scale.reciprocal().reshape(1)
                dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized_weight_tensor = (W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale)
        del W_float32, scale, U, Vh, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight_tensor

# --- Main script execution functions ---

def convert_to_fp8_scaled(
    input_file: str, output_file: str, comfy_quant: bool, t5xxl: bool, distillation_large: bool,
    distillation_small: bool, nerf_large: bool, nerf_small: bool,
    radiance: bool, wan: bool, qwen: bool, hunyuan: bool, zimage_l: bool, zimage_s: bool, calib_samples: int, seed: int,
    int8: bool = False, full_precision_matrix_mult: bool = False, skip_inefficient_layers: bool = False,
    include_input_scale: bool = False, no_learned_rounding: bool = False,
    **converter_kwargs
):
    # Determine target format
    target_format = 'int8' if int8 else 'fp8'
    target_dtype = TARGET_INT8_DTYPE if int8 else TARGET_FP8_DTYPE
    format_name = "INT8" if int8 else "FP8"
    
    print(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    print("-" * 60)
    if int8:
        print(f"Target format: INT8 (block-wise quantization)")
        print(f"INT8 Range: [{-INT8_SYMMETRIC_MAX}, {INT8_SYMMETRIC_MAX}]")
    else:
        print(f"Target FP8 format: {TARGET_FP8_DTYPE}\nFP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_device = device
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    if comfy_quant:
        print("Comfy quantization mode enabled: Using comfy_quant layer names and settings.")
        comfy_quant = True
    else:
        comfy_quant = False

    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device='cpu') as f:
            print(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        print(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Add target_format and no_learned_rounding to converter kwargs
    converter_kwargs['target_format'] = target_format
    converter_kwargs['no_learned_rounding'] = no_learned_rounding
    converter = LearnedRoundingConverter(**converter_kwargs)
    block_size = converter_kwargs.get('block_size', 64)

    print("\nScanning model and generating simulated calibration data...")
    calibration_data_cache = {}
    for key, tensor in tensors.items():
        if key.endswith('.weight') and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                print(f"  - Found new input dimension: {in_features}.")
                calibration_data_cache[in_features] = torch.randn(calib_samples, in_features, dtype=COMPUTE_DTYPE, generator=seed_generator, device=seed_device)
    print("Simulated calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight') and tensors[key].ndim == 2])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")
    print("-" * 60)

    for i, key in enumerate(weight_keys):
        skip_reason = ""

        if t5xxl and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            skipped_count += 1
            continue
        if t5xxl and any(n in key for n in AVOID_KEY_NAMES):
            skip_reason = "T5XXL exclusion"
        elif radiance and any(n in key for n in RADIANCE_LAYER_KEYNAMES):
            skip_reason = "Radiance exclusion"
        elif wan and any(n in key for n in AVOID_KEY_NAMES):
            skip_reason = "WAN exclusion"
        elif qwen and any(n in key for n in QWEN_AVOID_KEY_NAMES):
            skip_reason = "Qwen Image exclusion"
        elif zimage_l and any(n in key for n in ZIMAGE_AVOID_KEY_NAMES):
            skip_reason = "Z-Image exclusion"
        elif zimage_s and any(n in key for n in ZIMAGE_AVOID_KEY_NAMES):
            skip_reason = "Z-Image exclusion"
        elif hunyuan and any(n in key for n in HUNYUAN_AVOID_KEY_NAMES):
            skip_reason = "Hunyuan Video 1.5 exclusion"
        elif distillation_large and any(n in key for n in DISTILL_LAYER_KEYNAMES_LARGE):
            skip_reason = "Distillation layer and Flux1 keep in high precision"
        elif distillation_small and any(n in key for n in DISTILL_LAYER_KEYNAMES_SMALL):
            skip_reason = "Distillation layer keep in high precision"
        elif nerf_large and any(n in key for n in NERF_LAYER_KEYNAMES_LARGE):
            skip_reason = "NeRF layer, distillation layer and txt_in keep in high precision"
        elif nerf_small and any(n in key for n in NERF_LAYER_KEYNAMES_SMALL):
            skip_reason = "NeRF layer and distillation layer keep in high precision"
        elif wan and any(n in key for n in WAN_LAYER_KEYNAMES):
            skip_reason = "WAN layer keep in high precision"
        elif qwen and any(n in key for n in QWEN_LAYER_KEYNAMES):
            skip_reason = "Qwen Image layer keep in high precision"
        elif zimage_l and any(n in key for n in ZIMAGE_LAYER_KEYNAMES):
            skip_reason = "Z-Image layer keep in high precision"

        if skip_reason:
            print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
            original_tensor = tensors[key]
            new_tensors[key] = original_tensor.to(device='cpu', dtype=original_tensor.dtype)
            skipped_count += 1
            continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1
        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {key}")
            new_tensors[key] = original_tensor.to(device='cpu', dtype=original_tensor.dtype)
            continue
        
        # Check performance heuristics for inefficient layers
        if skip_inefficient_layers:
            should_skip, skip_perf_reason = should_skip_layer_for_performance(original_tensor, block_size)
            if should_skip:
                print(f"  - Skipping for performance: {skip_perf_reason}")
                new_tensors[key] = original_tensor.to(device='cpu', dtype=original_tensor.dtype)
                skipped_count += 1
                continue

        q_tensor, dequant_s, dequant_w = converter.convert(original_tensor)
        new_tensors[key] = q_tensor.to(device='cpu')
        base_name = key[:key.rfind('.weight')]
        bias_key = f"{base_name}.bias"
        
        if comfy_quant is True:
            new_tensors[f"{base_name}.weight_scale"] = dequant_s.to(device='cpu', dtype=SCALE_DTYPE).detach().clone()
            
            # Create appropriate comfy_quant tensor based on format
            if int8:
                comfy_quant_tensor = create_comfy_quant_tensor(
                    "int8_blockwise", 
                    block_size=block_size,
                    full_precision_matrix_mult=full_precision_matrix_mult if full_precision_matrix_mult else None
                )
                # Add input_scale placeholder for INT8 (required by ComfyUI)
                new_tensors[f"{base_name}.input_scale"] = torch.tensor([1.0], dtype=SCALE_DTYPE, device='cpu')
            else:
                comfy_quant_tensor = create_comfy_quant_tensor(
                    "float8_e4m3fn",
                    full_precision_matrix_mult=full_precision_matrix_mult if full_precision_matrix_mult else None
                )
                # Optionally add input_scale for FP8 (uses weight_scale as reasonable default)
                if include_input_scale:
                    new_tensors[f"{base_name}.input_scale"] = dequant_s.to(device='cpu', dtype=SCALE_DTYPE).detach().clone()
            new_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor.to(device='cpu')
        else:
            new_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device='cpu', dtype=SCALE_DTYPE).detach().clone()

        if bias_key in tensors:
            print(f"  - Adjusting corresponding bias: {bias_key}")
            with torch.no_grad():
                original_bias = tensors[bias_key]
                in_features = original_tensor.shape[1]
                if in_features not in calibration_data_cache:
                    print(f"  - WARNING: No calibration data for bias correction.")
                    new_tensors[bias_key] = original_bias
                else:
                    X_calib_dev = calibration_data_cache[in_features].to(device=device)
                    W_orig_dev = original_tensor.to(device=device, dtype=COMPUTE_DTYPE)
                    W_dequant_dev = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                    b_orig_dev = original_bias.to(device=device, dtype=COMPUTE_DTYPE)
                    weight_error = W_orig_dev - W_dequant_dev
                    output_error = X_calib_dev @ weight_error.T
                    bias_correction = output_error.mean(dim=0)
                    b_new = b_orig_dev - bias_correction
                    new_tensors[bias_key] = b_new.to(device='cpu', dtype=original_bias.dtype)
                    print(f"    - Original bias mean : {original_bias.mean().item():.6f}\n    - Corrected bias mean: {new_tensors[bias_key].mean().item():.6f}")
                    del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new
                    if device == 'cuda': torch.cuda.empty_cache()

        # T5XXL always needs input_scale regardless of --include_input_scale flag
        if t5xxl and f"{base_name}.input_scale" not in new_tensors:
            new_tensors[f"{base_name}.input_scale"] = dequant_s.to(device='cpu', dtype=SCALE_DTYPE).detach().clone()

        # Get scale key name based on comfy_quant mode
        scale_key = f"{base_name}.weight_scale" if comfy_quant else f"{base_name}.scale_weight"
        if scale_key in new_tensors:
            new_scale = new_tensors[scale_key]
            if dequant_s.ndim == 1:
                print(f"    - Final Dequant Scale value: {new_scale}\n    - Final Weight shape       : {q_tensor.shape}")
            else:
                print(f"    - Final Dequant Scale shape: {new_scale.shape}\n    - Final Weight shape       : {q_tensor.shape}")
        print("-" * 60)

    for key, tensor in tensors.items():
        if (any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and t5xxl):
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor

    if comfy_quant and "scaled_fp8" not in new_tensors and not int8:
        new_tensors["scaled_fp8"] = torch.empty((0), dtype=TARGET_FP8_DTYPE) if t5xxl else torch.empty((2), dtype=TARGET_FP8_DTYPE)

    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return

    print("-" * 60)
    print("Summary:")
    print(f"  - Original tensor count : {len(tensors)}\n  - Weights processed     : {processed_count}\n  - Weights skipped       : {skipped_count}\n  - Final tensor count    : {len(new_tensors)}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Convert safetensors weights to Scaled FP8 or INT8 format.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("-o", "--output", type=str, help="Output safetensors file path. Auto-generated if not provided.")
    parser.add_argument("--comfy_quant", action='store_true', help="Use Comfy quantization method.")
    parser.add_argument("--int8", action='store_true', help="Use INT8 block-wise quantization instead of FP8.")
    parser.add_argument("--kernel_backend", type=str, default="triton", choices=["triton", "triton_v2"],
                        help="Kernel backend for INT8 quantization. 'triton' uses quant_ops.py, 'triton_v2' uses kernels/int8_matmul.py with autotuned kernels.")
    parser.add_argument("--simple", action='store_true', help="Skip SVD optimization, use simple quantization.")
    parser.add_argument("--full_precision_matrix_mult", action='store_true', help="Add full_precision_matrix_mult=True to .comfy_quant metadata.")
    parser.add_argument("--heur", action='store_true', help="Skip layers with poor quantization characteristics (aspect ratio, size).")
    parser.add_argument("--input_scale", action='store_true', help="Include input_scale tensor for FP8 (uses weight_scale as default). Always enabled for T5XXL.")
    parser.add_argument("--t5xxl", action='store_true', help="Apply exclusions for T5XXL Text Encoder models.")
    parser.add_argument("--distillation_large", action='store_true', help="Exclude known distillation layers and other sensitive.")
    parser.add_argument("--distillation_small", action='store_true', help="Exclude known distillation layers.")
    parser.add_argument("--nerf_large", action='store_true', help="Exclude known NeRF layers, distillation layers and txt_in.")
    parser.add_argument("--nerf_small", action='store_true', help="Exclude known NeRF layers and distillation layers.")
    parser.add_argument("--radiance", action='store_true', help="Exclude known Radiance Field layers.")
    parser.add_argument("--wan", action='store_true', help="Exclude known WAN layers.")
    parser.add_argument("--qwen", action='store_true', help="Exclude known Qwen Image layers.")
    parser.add_argument("--hunyuan", action='store_true', help="Exclude known Hunyuan Video 1.5 layers.")
    parser.add_argument("--zimage_l", action='store_true', help="Exclude known Z-Image layers.")
    parser.add_argument("--zimage_s", action='store_true', help="Exclude known Z-Image layers.")
    parser.add_argument("--full_matrix", action='store_true', help="If should use torch.linalg.svd with full matices instead of the torch.svd_lowrank.")
    parser.add_argument("--scaling_mode", type=str, default="tensor", choices=["tensor", "block"], help="Quantization scaling mode.")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for 'block' scaling mode (used by both FP8 and INT8).")
    parser.add_argument("--calib_samples", type=int, default=3072, help="Number of random samples for bias correction.")
    parser.add_argument("--manual_seed", type=int, default=-1, help="Set a manual seed for reproducibility. Use -1 for random.")
    parser.add_argument("--optimizer", type=str, default="original", choices=["original", "adamw", "ppsf", "radam"], help="Optimization algorithm.")
    parser.add_argument("--num_iter", type=int, default=500, help="Total optimization iterations per tensor.")
    parser.add_argument("--lr", type=float, default=1e-2, help="[AdamW/RAdam/Original] Initial learning rate.")
    parser.add_argument("--top_p", type=float, default=0.01, help="Proportion of principal components (SVD) to use.")
    parser.add_argument("--min_k", type=int, default=1, help="Minimum number of principal components.")
    parser.add_argument("--max_k", type=int, default=16, help="Maximum number of principal components.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Check triton_v2 kernel backend availability
    if args.kernel_backend == 'triton_v2' and not _HAS_TRITON_V2:
        print("ERROR: triton_v2 kernel backend requested but not available.")
        print("       The kernels/int8_matmul.py module could not be imported.")
        print("")
        print("Fallback command: Remove '--kernel_backend triton_v2' or use '--kernel_backend triton'")
        sys.exit(1)

    # Only check FP8 support if not using INT8
    if not args.int8:
        try:
            _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE, device='cuda' if torch.cuda.is_available() else 'cpu')
        except (RuntimeError, TypeError):
            print("Error: This hardware/PyTorch version does not support the target FP8 dtype.")
            return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        if args.int8:
            format_str = "int8_blockwise"
            scaling_str = f"_bs{args.block_size}"
        else:
            format_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
            scaling_str = f"_{args.scaling_mode}"
        flags = "".join(["_t5" if args.t5xxl else "", "_nodist_l" if args.distillation_large else "", "_nodist_s" if args.distillation_small else "", "_nonerf_l" if args.nerf_large else "", "_nonerf_s" if args.nerf_small else "", "_norad" if args.radiance else ""])
        output_file = f"{base}_{format_str}{scaling_str}{flags}_k{args.min_k}-{args.max_k}_p{args.top_p}_lr{args.lr}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be same as input.")
        return

    seed = int(torch.randint(0, 2**32 - 1, ()).item()) if args.manual_seed == -1 else args.manual_seed
    print(f"Using seed: {seed}")

    # Separate converter kwargs from function kwargs
    excluded_keys = ['input', 'output', 'comfy_quant', 't5xxl', 'distillation_large', 'distillation_small', 
                     'nerf_large', 'nerf_small', 'radiance', 'wan', 'qwen', 'hunyuan', 'zimage_l', 'zimage_s', 
                     'calib_samples', 'manual_seed', 'int8', 'full_precision_matrix_mult', 'heur',
                     'input_scale', 'simple']
    converter_kwargs = {k: v for k, v in vars(args).items() if k not in excluded_keys}

    convert_to_fp8_scaled(
        args.input, output_file, args.comfy_quant, args.t5xxl, args.distillation_large,
        args.distillation_small, args.nerf_large, args.nerf_small,
        args.radiance, args.wan, args.qwen, args.hunyuan, args.zimage_l, args.zimage_s, args.calib_samples, seed,
        int8=args.int8, full_precision_matrix_mult=args.full_precision_matrix_mult,
        skip_inefficient_layers=args.heur,
        include_input_scale=args.input_scale,
        no_learned_rounding=args.simple,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
