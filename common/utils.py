import torch as th
import torch.nn.functional as F
from typing import List
import numpy as np


def pad_tensor(tensor, max_length, dim, pad_value=0):
    if dim < 0:
        dim += tensor.ndim
    pad = max_length - tensor.shape[dim]
    if pad <= 0:
        return tensor
    # 2. Construct the padding tuple
    #    F.pad expects: (last_left, last_right, 2nd_last_left, 2nd_last_right, ...)
    #    We need 2 padding values (left, right) per dimension.
    #    So the total tuple length is 2 * ndim.
    pad_list = [0] * (2 * tensor.ndim)
    
    # 3. Calculate index
    #    The "right" (after) pad for dim 'd' is at index: 2 * (ndim - 1 - d) + 1
    #    The "left" (before) pad for dim 'd' is at index: 2 * (ndim - 1 - d)
    idx_for_padding_after = 2 * (tensor.ndim - 1 - dim) + 1
    
    pad_list[idx_for_padding_after] = pad
    return F.pad(tensor, tuple(pad_list), mode='constant', value=pad_value)

def pad_repeat_tensor(x, lengths, max_len, pad_dim):
        """
        Extends a tensor along an arbitrary dimension 'pad_dim' using repetition padding.
        
        Args:
            x: Input tensor of shape (Batch, ..., T, ...).
            lengths: IntTensor of shape (Batch,) containing valid lengths.
            max_len: Target size for pad_dim.
            pad_dim: The dimension index to pad. (Cannot be 0, which is Batch).
            
        Returns:
            Tensor with pad_dim size = max_len.
        """
        # 1. Validation
        if pad_dim == 0:
            raise ValueError("pad_dim cannot be 0 (batch dimension).")
        
        # 2. Transpose pad_dim to dimension 1
        # This standardizes the shape to (Batch, T, ...others...)
        # .contiguous() is required because transpose creates a view with non-standard strides,
        # but .view() (used later) requires contiguous memory.
        x_transposed = x.transpose(1, pad_dim).contiguous()
        
        # Capture shape details
        B, T = x_transposed.shape[:2]
        other_dims = x_transposed.shape[2:]
        
        # 3. Flatten "other dimensions" into a single feature dimension
        # Shape: (B, T, Features)
        x_flat = x_transposed.view(B, T, -1)
        features_size = x_flat.size(2)
        
        # --- Core Padding Logic (Same as before) ---
        
        # Prepare Output
        out_flat = th.empty((B, max_len, features_size), dtype=x.dtype, device=x.device)
        
        # Copy valid initial data
        copy_lim = min(T, max_len)
        out_flat[:, :copy_lim, :] = x_flat[:, :copy_lim, :]
        
        # Gather last valid frames
        batch_indices = th.arange(B, device=x.device)
        # Clamp to prevent out-of-bounds if lengths are wrong
        safe_last_indices = (lengths - 1).clamp(min=0, max=T-1).long() 
        
        # Extract last frames: (B, Features)
        last_frames = x_flat[batch_indices, safe_last_indices]
        
        # Create Mask: (B, max_len)
        seq_range = th.arange(max_len, device=x.device).unsqueeze(0)
        lengths_col = lengths.unsqueeze(1)
        padding_mask = seq_range >= lengths_col
        
        # Expand for broadcasting
        # Mask: (B, max_len, 1) | Values: (B, 1, Features)
        padding_mask = padding_mask.unsqueeze(-1)
        last_frames_expanded = last_frames.unsqueeze(1)
        
        # Fill padding zones
        out_flat = th.where(padding_mask, last_frames_expanded, out_flat)
        
        # --- End Core Logic ---
        
        # 4. Reshape back to original structure
        # View as (B, max_len, d2, d3...)
        new_shape = (B, max_len) + other_dims
        out_transposed = out_flat.view(new_shape)
        
        # 5. Transpose back to original dimension order
        out = out_transposed.transpose(1, pad_dim)
        
        return out


def repeat_last_valid(tensor: th.Tensor, lengths: th.Tensor, dim: int) -> th.Tensor:
    """
    Repeats the last valid value along a dimension based on a lengths tensor.
    
    Args:
        tensor: The input tensor of shape (..., D, ...).
        lengths: A tensor containing the lengths of valid data. 
                 Usually shape (batch_size,), but must be broadcastable 
                 to the tensor's shape excluding 'dim'.
        dim: The dimension along which to repeat the values.
        
    Returns:
        A tensor with the same shape as input, where values after the length
        boundary are repeats of the value at (length - 1).
    """
    ndim = tensor.ndim
    
    # Handle negative dimension indexing (e.g., -1)
    if dim < 0:
        dim += ndim

    # 1. Create a range of indices along the target dimension
    # Shape: (1, 1, ..., size(dim), ..., 1)
    max_len = tensor.size(dim)
    idx_shape = [1] * ndim
    idx_shape[dim] = max_len
    idx_range = th.arange(max_len, device=tensor.device).view(idx_shape)
    
    # 2. Reshape lengths to ensure correct broadcasting
    # We need 'lengths' to have 1s in the target dim and all trailing dims
    # e.g., if Tensor is (B, T, D) and L is (B), we want L to be (B, 1, 1)
    lens_view = lengths.unsqueeze(dim)
    
    # Pad trailing dimensions with 1s until ranks match
    while lens_view.ndim < ndim:
        lens_view = lens_view.unsqueeze(-1)
        
    # 3. Compute Gather Indices
    # Logic: target_idx = min(current_idx, length - 1)
    # This clamps the index to the last valid position.
    target_indices = th.min(idx_range, lens_view - 1)
    
    # Expand indices to match the full tensor shape for gather
    target_indices = target_indices.expand_as(tensor)
    
    # 4. Gather the values
    return th.gather(tensor, dim, target_indices)


def pad_and_stack_numpy(batch_of_lists: List[List[np.ndarray]]):
    """
    Pads a list of lists of 1D numpy arrays into a (K, T, D) array.
    
    Args:
        batch_of_lists: List[List[np.ndarray]]
                        Outer list is batch (K).
                        Inner list is sequence (T_var).
                        Arrays are variable length vectors (D_var).
                        
    Returns:
        np.ndarray: Shape (K, T_max, D_max)
    """
    # 1. Calculate Target Dimensions
    # T_max: The longest sequence in the batch
    max_t = max(len(seq) for seq in batch_of_lists)
    
    # D_max: The longest vector across all sequences in the batch
    max_d = max(max(vec.shape[0] for vec in seq) for seq in batch_of_lists)

    processed_sequences = []

    for seq in batch_of_lists:
        # --- Stage 1: Pad Feature Dimension (D) ---
        padded_vectors = []
        for vec in seq:
            current_d = vec.shape[0]
            pad_amt_d = max_d - current_d
            
            if pad_amt_d > 0:
                # Repeat the last value of the vector
                # np.repeat with a scalar repeats that value N times
                padding = np.repeat(vec[-1], pad_amt_d)
                vec_padded = np.concatenate([vec, padding])
            else:
                vec_padded = vec
            
            padded_vectors.append(vec_padded)
        
        # Stack vectors to form a sequence array of shape (Current_T, Max_D)
        seq_array = np.stack(padded_vectors)

        # --- Stage 2: Pad Time Dimension (T) ---
        current_t = seq_array.shape[0]
        pad_amt_t = max_t - current_t
        
        if pad_amt_t > 0:
            # Repeat the last vector in the sequence
            # np.newaxis is the equivalent of unsqueeze(0)
            last_vector = seq_array[-1][np.newaxis, :] 
            
            # np.tile is the closest equivalent to torch.repeat() for tiling blocks
            # Tile pad_amt_t times along axis 0, and 1 time along axis 1
            padding_block = np.tile(last_vector, (pad_amt_t, 1))
            
            seq_array_padded = np.concatenate([seq_array, padding_block], axis=0)
        else:
            seq_array_padded = seq_array

        processed_sequences.append(seq_array_padded)

    # --- Final Stack ---
    # Stack all processed sequences along the batch dimension
    return np.stack(processed_sequences)

